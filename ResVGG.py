import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== 设备设置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ====================== 超参数 ======================
np.random.seed(42)
torch.manual_seed(42)

IMG_H, IMG_W = 256, 256
OUT_DIM = 4          # [x, y, h, SWA]
EPOCHS = 200
BATCH_SIZE = 32
LR_PHASE1 = 1e-4
LR_PHASE2 = 5e-5
MIN_LR = 1e-6
DIAG_THRESHOLD = 0.5
UNCERTAINTY_WEIGHT = 0.2
DYNAMIC_WEIGHT_INIT_CLS = 0.8
DYNAMIC_WEIGHT_FINAL_CLS = 0.2
DYNAMIC_WARMUP_EPOCHS = 100
PHASE1_TARGET_ACC = 0.99

DATA_PATH = "data/uda_dataset.npz"

PARAM_NAMES = ["x", "y", "h", "SWA"]


# ====================== 数据加载 ======================
def load_dataset():
    """
    从 npz 文件加载真实 EUV 仿真数据。

    原始标签列：[x, y, h, LS_flag, SWA]（5列）
    处理逻辑：
      - LS_flag (col3) → 转 one-hot [1,0]/[0,1]，作为域条件注入特征层，不进输出头
      - 输出标签统一为 4 维 [x, y, h, SWA]
      - 光板 (LS_flag=0) 的 SWA 回归值强制置 0.0，分类标签强制置 0（占位符策略）
      - 回归标签按 cls=1 有效样本的值域归一化到 [0,1]，返回 reg_min/reg_max 供反归一化
    """
    data = np.load(DATA_PATH)
    images  = data['images'].astype(np.float32)       # (N, 256, 256)
    reg_raw = data['labels_reg'].astype(np.float32)   # (N, 5)
    cls_raw = data['labels_cls'].astype(np.int64)     # (N, 5)

    ls_flag = reg_raw[:, 3].astype(np.int64)          # 0=光板, 1=有图形

    # one-hot 条件向量：光板=[1,0], 有图形=[0,1]
    ls_onehot = np.zeros((len(ls_flag), 2), dtype=np.float32)
    ls_onehot[ls_flag == 0, 0] = 1.0
    ls_onehot[ls_flag == 1, 1] = 1.0

    # 4 维标签 [x, y, h, SWA]
    y_reg = np.concatenate([reg_raw[:, :3], reg_raw[:, 4:5]], axis=1).copy()
    y_cls = np.concatenate([cls_raw[:, :3], cls_raw[:, 4:5]], axis=1).astype(np.float32)

    # 占位符策略：光板数据的 SWA 强制清零
    blank_mask = (ls_flag == 0)
    y_reg[blank_mask, 3] = 0.0
    y_cls[blank_mask, 3] = 0.0

    # 用 cls=1 有效样本确定各列归一化范围，使回归目标落入 [0,1]
    reg_min = np.zeros(OUT_DIM, dtype=np.float32)
    reg_max = np.ones(OUT_DIM,  dtype=np.float32)
    for i in range(OUT_DIM):
        valid_vals = y_reg[y_cls[:, i] == 1, i]
        if len(valid_vals) > 0:
            reg_min[i] = valid_vals.min()
            reg_max[i] = valid_vals.max()

    # 归一化（占位符 0.0 在归一化后变为负数，但会被掩码截断，不影响训练）
    denom = (reg_max - reg_min).clip(min=1e-6)
    y_reg_norm = (y_reg - reg_min) / denom
    # 占位符位置归一化后强制回 0，保证掩码截断效果纯净
    y_reg_norm[blank_mask, 3] = 0.0

    # 图像加通道维度 (N,1,H,W)，per-dataset 标准化
    x = images[:, np.newaxis, :, :]
    m, s = x.mean(), x.std() + 1e-6
    x = (x - m) / s

    return x, y_cls, y_reg_norm, ls_onehot, reg_min, reg_max


class LithoDataset(Dataset):
    def __init__(self, x, d, e, ls):
        self.x  = torch.tensor(x)
        self.d  = torch.tensor(d)
        self.e  = torch.tensor(e)
        self.ls = torch.tensor(ls)   # (N, 2) one-hot 域条件

    def __len__(self): return len(self.x)

    def __getitem__(self, idx): return self.x[idx], self.d[idx], self.e[idx], self.ls[idx]


# ====================== 模型架构 ======================
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False), nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.last_sa_map = None  # 用于可视化空间注意力

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        self.last_sa_map = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * self.last_sa_map


class ResNetBottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes * 4))
        self.cbam = CBAM(planes * 4) if use_cbam else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn3(self.conv3(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))) + self.shortcut(x))
        return self.cbam(out)


class EncoderBackbone(nn.Module):
    def __init__(self, use_cbam=True):
        super().__init__()
        self.use_cbam = use_cbam
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        # ResNet 路：第3/4 bottleneck 在高分辨率(25x25/13x13)加 CBAM
        self.res1 = ResNetBottleneck(64, 64, 1, use_cbam=False)
        self.res2 = ResNetBottleneck(256, 128, 2, use_cbam=False)
        self.res3 = ResNetBottleneck(512, 256, 2, use_cbam=use_cbam)
        self.res4 = ResNetBottleneck(1024, 512, 2, use_cbam=use_cbam)
        # VGG 路：两个 MaxPool 后各插一个 CBAM
        self.vgg_block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.vgg_cbam1 = CBAM(128) if use_cbam else nn.Identity()
        self.vgg_block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.vgg_cbam2 = CBAM(256) if use_cbam else nn.Identity()
        self.vgg_block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
        )
        # 两路各自 global pool 后在特征维 concat，再用 1x1 Linear 融合，输出 2560
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 512, 2048 + 512, bias=False),
            nn.BatchNorm1d(2048 + 512), nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, ls_onehot):
        stem_out = self.stem(x)
        # ResNet 路 → global pool
        res_out = self.global_pool(self.res4(self.res3(self.res2(self.res1(stem_out))))).flatten(1)  # (B,2048)
        # VGG 路 → global pool
        vgg_out = self.global_pool(self.vgg_block3(self.vgg_cbam2(self.vgg_block2(self.vgg_cbam1(self.vgg_block1(stem_out)))))).flatten(1)  # (B,512)
        img_feat = self.fusion(torch.cat([res_out, vgg_out], dim=1))    # (B, 2560)
        # 条件注入：拼接 LS 域 one-hot，赋予网络"硬开关"感知物理背景
        return torch.cat([img_feat, ls_onehot], dim=1)                  # (B, 2562)


class ResVGG_DualBranch(nn.Module):
    def __init__(self, use_cbam=True):
        super().__init__()
        self.encoder = EncoderBackbone(use_cbam)
        feat_dim = 2562  # 2560 图像特征 + 2 LS one-hot
        self.diag_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(True), nn.Dropout(0.3),
                                       nn.Linear(512, OUT_DIM))   # 输出 logits，配合 BCEWithLogitsLoss
        self.reg_head = nn.Sequential(nn.Linear(feat_dim + OUT_DIM, 512), nn.ReLU(True), nn.Dropout(0.3),
                                      nn.Linear(512, OUT_DIM))
        self.uncert_head = nn.Sequential(nn.Linear(feat_dim + OUT_DIM, 512), nn.ReLU(True),
                                         nn.Linear(512, OUT_DIM))

    def forward(self, x, ls_onehot, detach_diag_for_reg=False):
        feat = self.encoder(x, ls_onehot)                                   # (B, 2562)
        diag_logits = self.diag_head(feat)                                  # logits，供 BCEWithLogitsLoss
        diag_prob   = torch.sigmoid(diag_logits)                            # 概率，供门控和评估
        reg_signal  = diag_prob.detach() if detach_diag_for_reg else diag_prob
        # warm-up 阶段用 hard gate 防止软概率把回归梯度压死
        hard_gate   = (diag_prob.detach() >= DIAG_THRESHOLD).float()
        gate        = hard_gate if not self.training else reg_signal        # 训练时用软概率，评估时已是 eval
        esti_in     = torch.cat([feat, reg_signal], dim=1)
        return diag_logits, torch.sigmoid(self.reg_head(esti_in)) * gate, \
               (F.softplus(self.uncert_head(esti_in)) + 1e-4) * gate + 1e-4


class DirectRegResVGG(nn.Module):
    """消融实验Baseline1：无门控单分支直接回归网络"""

    def __init__(self):
        super().__init__()
        self.encoder = EncoderBackbone(use_cbam=False)
        self.reg_head = nn.Sequential(nn.Linear(2562, 512), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(512, OUT_DIM))

    def forward(self, x, ls_onehot):
        return None, torch.sigmoid(self.reg_head(self.encoder(x, ls_onehot))), None


# ====================== 损失与训练逻辑 ======================
def soft_masked_mae_loss(pred, target, soft_mask): return (
            torch.abs(pred - target) * soft_mask.clamp(0, 1)).sum() / soft_mask.sum().clamp_min(1e-6)


def soft_masked_uncertainty_mae(pred_sigma, pred_mean, target, soft_mask): return (
            torch.abs(pred_sigma - torch.abs(pred_mean.detach() - target)) * soft_mask.clamp(0,
                                                                                             1)).sum() / soft_mask.sum().clamp_min(
    1e-6)


def get_dynamic_loss_weights(epoch):
    if epoch < DYNAMIC_WARMUP_EPOCHS: return DYNAMIC_WEIGHT_INIT_CLS, 1.0 - DYNAMIC_WEIGHT_INIT_CLS
    ratio = (epoch - DYNAMIC_WARMUP_EPOCHS) / (EPOCHS - DYNAMIC_WARMUP_EPOCHS)
    cls_w = max(DYNAMIC_WEIGHT_FINAL_CLS,
                DYNAMIC_WEIGHT_INIT_CLS - ratio * (DYNAMIC_WEIGHT_INIT_CLS - DYNAMIC_WEIGHT_FINAL_CLS))
    return cls_w, 1.0 - cls_w


def train_model(model, train_loader, val_loader, reg_min, reg_max, model_name="Dual_CBAM", is_dual=True):
    optimizer = optim.Adam(model.parameters(), lr=LR_PHASE1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    # pos_weight 按列不平衡比计算：x/y/h ≈2.27，SWA ≈12.0
    pos_w = torch.tensor([2.27, 2.27, 2.27, 12.0], device=device)
    bce_logits = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    mse = nn.MSELoss()

    for ep in range(EPOCHS):
        model.train()
        cls_w, reg_w = get_dynamic_loss_weights(ep) if is_dual else (0, 1.0)

        for x, d, e, ls in train_loader:
            x, d, e, ls = x.to(device), d.to(device), e.to(device), ls.to(device)
            optimizer.zero_grad()
            if is_dual:
                diag_logits, pe, sigma = model(x, ls)
                diag_prob = torch.sigmoid(diag_logits.detach())
                loss = cls_w * bce_logits(diag_logits, d) + reg_w * (
                    soft_masked_mae_loss(pe, e, diag_prob) +
                    UNCERTAINTY_WEIGHT * soft_masked_uncertainty_mae(sigma, pe, e, diag_prob))
            else:
                _, pe, _ = model(x, ls)
                loss = mse(pe, e)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (ep + 1) % 10 == 0:
            val_acc, val_mae = evaluate_model(model, val_loader, reg_min, reg_max, is_dual)
            print(f"[{model_name}] Epoch {ep + 1:4d} | Acc: {val_acc:.4f} | MAE: {val_mae:.4f}")
    return evaluate_model(model, val_loader, reg_min, reg_max, is_dual)


def evaluate_model(model, loader, reg_min, reg_max, is_dual=True):
    """按列分别反归一化后计算 MAE，再对有效列平均（物理单位：nm / 度）"""
    model.eval()
    t_min = torch.tensor(reg_min, device=device)                         # (4,)
    t_rng = torch.tensor(reg_max - reg_min, device=device).clamp(1e-6)  # (4,)

    col_mae   = torch.zeros(OUT_DIM, device=device)  # 各列累计绝对误差
    col_count = torch.zeros(OUT_DIM, device=device)  # 各列有效样本计数
    accs = []

    with torch.no_grad():
        for x, d, e, ls in loader:
            x, d, e, ls = x.to(device), d.to(device), e.to(device), ls.to(device)
            if is_dual:
                diag_logits, pe, _ = model(x, ls)
                diag_prob = torch.sigmoid(diag_logits)
                pred_cls  = (diag_prob >= DIAG_THRESHOLD).float()
                accs.append((pred_cls == d).float().mean().item())
                # 按列掩码：真实标签为 1 的位置才纳入 MAE 统计
                col_mask = (d > DIAG_THRESHOLD)                          # (B, 4) bool
            else:
                _, pe, _ = model(x, ls)
                accs.append(0.0)
                col_mask = (e > 0)                                       # (B, 4) bool

            # 反归一化：各列独立缩放
            pe_phys = pe * t_rng + t_min   # (B, 4)
            e_phys  = e  * t_rng + t_min   # (B, 4)
            abs_err = torch.abs(pe_phys - e_phys)                        # (B, 4)
            col_mae   += (abs_err * col_mask.float()).sum(dim=0)
            col_count += col_mask.float().sum(dim=0)

    # 对有效列（至少有1个样本）取均值，再平均
    valid_cols = col_count > 0
    per_col_mae = torch.where(valid_cols, col_mae / col_count.clamp(1), torch.zeros_like(col_mae))
    mean_mae = per_col_mae[valid_cols].mean().item()
    return np.mean(accs) if is_dual else 0.0, mean_mae


# ====================== 【新增核心绘图函数】 ======================

def plot_confusion_matrix(model, loader):
    """绘制诊断分支多标签预测混淆矩阵"""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, d, _, ls in loader:
            diag_logits, _, _ = model(x.to(device), ls.to(device))
            pd = torch.sigmoid(diag_logits)
            all_true.append(d.cpu().numpy())
            all_pred.append((pd.cpu().numpy() > DIAG_THRESHOLD).astype(int))
    all_true = np.concatenate(all_true).flatten()
    all_pred = np.concatenate(all_pred).flatten()

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Defected"], yticklabels=["Normal", "Defected"])
    plt.title("Diagnosis Branch Confusion Matrix\n(EUV Multi-parameter Decoupling)")
    plt.ylabel('True Status')
    plt.xlabel('Predicted Status')
    plt.tight_layout()
    plt.savefig('fig_confusion_matrix.png', dpi=300)
    plt.close()
    print("已生成混淆矩阵: fig_confusion_matrix.png")


def plot_ablation_study(metrics_dict):
    """绘制消融实验双轴柱状图"""
    labels = ['Baseline\n(Direct Reg)', 'Dual-Branch\n(No CBAM)', 'Proposed\n(Dual + CBAM)']
    acc_data = [metrics_dict['Base'][0], metrics_dict['Dual_NoCBAM'][0], metrics_dict['Dual_CBAM'][0]]
    mae_data = [metrics_dict['Base'][1], metrics_dict['Dual_NoCBAM'][1], metrics_dict['Dual_CBAM'][1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar1 = ax1.bar(x - width / 2, acc_data, width, label='Diagnosis Acc (%)', color='#4C72B0')
    ax1.set_ylabel('Accuracy (%)', color='#4C72B0', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width / 2, mae_data, width, label='Estimation MAE', color='#C44E52')
    ax2.set_ylabel('Mean Absolute Error (MAE)', color='#C44E52', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    plt.title('Ablation Study of Proposed Modules on EUV Defect Inversion', fontsize=14)

    # 图例
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig('fig_ablation_study.png', dpi=300)
    plt.close()
    print("已生成消融实验柱状图: fig_ablation_study.png")


def plot_attention_map(model, test_img, test_ls):
    """注意力机制可视化 (物理敏感区域)"""
    model.eval()
    img_tensor = torch.tensor(test_img).to(device)
    ls_tensor  = torch.tensor(test_ls).to(device)
    with torch.no_grad():
        _ = model(img_tensor, ls_tensor)
        sa_map = model.encoder.res4.cbam.last_sa_map.cpu().numpy()[0, 0]

    # 将 13x13 热力图插值放大到 200x200
    sa_map_resized = torch.nn.functional.interpolate(
        torch.from_numpy(sa_map).float().unsqueeze(0).unsqueeze(0),
        size=(IMG_H, IMG_W), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # test_img 形状为 (1, 1, H, W)，取 [0, 0] 得到 (H, W) 供 imshow 使用
    display_img = test_img[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title("Input EUV Difference Image")
    axes[0].axis('off')

    axes[1].imshow(sa_map_resized, cmap='jet')
    axes[1].set_title("CBAM Spatial Attention Heatmap")
    axes[1].axis('off')

    axes[2].imshow(display_img, cmap='gray')
    axes[2].imshow(sa_map_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay (Focus on Defect Physics)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('fig_attention_visualization.png', dpi=300)
    plt.close()
    print("已生成注意力机制可视化图: fig_attention_visualization.png")


def main():
    print("1. 加载真实 EUV 仿真数据集...")
    x, d, e, ls, reg_min, reg_max = load_dataset()
    print(f"   数据量: {len(x)}, 图像尺寸: {x.shape[2]}x{x.shape[3]}, 输出维度: {OUT_DIM}")
    print(f"   回归值域 (物理单位): {dict(zip(PARAM_NAMES, [f'{reg_min[i]:.2f}~{reg_max[i]:.2f}' for i in range(OUT_DIM)]))}")

    xt, xv, dt, dv, et, ev, lt, lv = train_test_split(
        x, d, e, ls, test_size=0.2, random_state=42)
    train_loader = DataLoader(LithoDataset(xt, dt, et, lt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(LithoDataset(xv, dv, ev, lv), batch_size=BATCH_SIZE)

    metrics = {}

    # === 模型 A: 传统直接回归 Baseline ===
    print("\n2. 训练传统 Baseline 模型 (Direct Regression)...")
    model_base = DirectRegResVGG().to(device)
    acc_base, mae_base = train_model(model_base, train_loader, val_loader, reg_min, reg_max, "Base", is_dual=False)
    metrics['Base'] = (0.0, mae_base)

    # === 模型 B: 双分支 (无注意力) ===
    print("\n3. 训练双分支消融模型 (Dual-Branch without CBAM)...")
    model_dual_no_cbam = ResVGG_DualBranch(use_cbam=False).to(device)
    acc_d1, mae_d1 = train_model(model_dual_no_cbam, train_loader, val_loader, reg_min, reg_max, "Dual_NoCBAM", is_dual=True)
    metrics['Dual_NoCBAM'] = (acc_d1, mae_d1)

    # === 模型 C: 提出模型 (双分支 + CBAM) ===
    print("\n4. 训练提出模型 (Proposed Dual-Branch + CBAM)...")
    model_dual_cbam = ResVGG_DualBranch(use_cbam=True).to(device)
    acc_d2, mae_d2 = train_model(model_dual_cbam, train_loader, val_loader, reg_min, reg_max, "Dual_CBAM", is_dual=True)
    metrics['Dual_CBAM'] = (acc_d2, mae_d2)

    print("\n================ 综合性能测试报告 ================")
    print(f"Baseline 直接回归  | MAE: {mae_base:.4f} nm/°")
    print(f"双分支无注意力     | Acc: {acc_d1:.4f}, MAE: {mae_d1:.4f} nm/°")
    print(f"提出模型(完整态)   | Acc: {acc_d2:.4f}, MAE: {mae_d2:.4f} nm/°")
    print("==================================================")

    # 生成论文图表
    print("\n5. 正在生成科研图表...")
    plot_confusion_matrix(model_dual_cbam, val_loader)
    plot_ablation_study(metrics)

    # 取验证集第一张样本测试注意力机制
    test_img = xv[0:1]
    test_ls  = lv[0:1]
    plot_attention_map(model_dual_cbam, test_img, test_ls)
    print("全部流程执行完毕！")


if __name__ == "__main__":
    main()