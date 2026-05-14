import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ====================== 设备设置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ====================== 超参数 (保持原样) ======================
np.random.seed(42)
torch.manual_seed(42)

IMG_H, IMG_W = 200, 200
OUT_DIM = 12
N_SAMPLES = 1000
EPOCHS = 200
BATCH_SIZE = 32
LR_PHASE1 = 1e-4
LR_PHASE2 = 5e-5
MIN_LR = 1e-6
MAX_VAL = 0.983330
MIN_VAL = 0.0
MAX_BIAS_ABS = 0.6
DECIMAL_DIGIT = 6
DIAG_THRESHOLD = 0.5
UNCERTAINTY_WEIGHT = 0.2
DYNAMIC_WEIGHT_INIT_CLS = 0.8
DYNAMIC_WEIGHT_FINAL_CLS = 0.2
DYNAMIC_WARMUP_EPOCHS = 100
PHASE1_TARGET_ACC = 0.99

DIFF_MEAN = 0.0
DIFF_STD = 1.0

PARAM_NAMES = [
    "P0:Blur", "P1:H-Grad", "P2:V-Grad", "P3:Lum",
    "P4:Contrast", "P5:Noise", "P6:H-Shift", "P7:V-Shift",
    "P8:Mask", "P9:Edge", "P10:Grid", "P11:Cross"
]


# ====================== 数据集生成 (保持原样) ======================
def round_clip(arr): return np.round(np.clip(arr, MIN_VAL, MAX_VAL), DECIMAL_DIGIT)


def create_cross_background(h, w, max_val=MAX_VAL):
    img = np.full((h, w), 0.200000, dtype=np.float64)
    cross_w = 12
    center_h, center_w = h // 2, w // 2
    img[center_h - cross_w // 2:center_h + cross_w // 2, :] = max_val
    img[:, center_w - cross_w // 2:center_w + cross_w // 2] = max_val
    return round_clip(img)


base_img = create_cross_background(IMG_H, IMG_W, MAX_VAL)


def add_distortion(img, param_idx, bias):
    strength = abs(float(bias))
    sign = 1.0 if bias >= 0 else -1.0
    res = img.copy()
    if param_idx == 0:
        res = gaussian_filter(res, sigma=max(0.2, strength * 1.2)) if sign > 0 else res + (
                    res - gaussian_filter(res, sigma=1.0)) * strength * 0.8
    elif param_idx == 1:
        x, y = np.linspace(-1, 1, IMG_W), np.linspace(-1, 1, IMG_H)
        xx, yy = np.meshgrid(x, y)
        res += (xx ** 2 + yy ** 2) * sign * strength * 0.12
    elif param_idx == 2:
        res += np.linspace(0, sign * strength, IMG_H)[:, np.newaxis].repeat(IMG_W, axis=1) * 0.08
    elif param_idx == 3:
        res += sign * strength * 0.06
    elif param_idx == 4:
        blur = gaussian_filter(res, sigma=1.0)
        res = res - blur * strength * 0.1 if sign > 0 else res + (res - blur) * strength * 0.3
    elif param_idx == 5:
        res += (np.random.uniform(-0.02, 0.02, size=img.shape) * strength + sign * strength * 0.005)
    elif param_idx == 6:
        shift = max(1, int(np.ceil(strength * 10)))
        s = shift if sign > 0 else -shift
        new_res = np.zeros_like(res)
        if s > 0:
            new_res[:, s:] = res[:, :-s]
        else:
            new_res[:, :s] = res[:, -s:]
        res = new_res
    elif param_idx == 7:
        res = np.roll(res, max(1, int(np.ceil(strength * 8))) * (1 if sign > 0 else -1), axis=0)
    elif param_idx == 8:
        mask = np.zeros_like(res);
        mask[80:120, 80:120] = 1.0
        res += (-sign) * mask * strength * 0.07
    elif param_idx == 9:
        edge_scale = 1 - sign * strength * 0.12
        res[:30, :] *= edge_scale;
        res[-30:, :] *= edge_scale
        res[:, :30] *= edge_scale;
        res[:, -30:] *= edge_scale
    elif param_idx == 10:
        gx, gy = np.meshgrid(np.linspace(0, sign * strength * 0.1, IMG_W), np.linspace(0, sign * strength * 0.1, IMG_H))
        res += gx + gy
    elif param_idx == 11:
        center_h, center_w = IMG_H // 2, IMG_W // 2
        width = 12 + max(1, int(np.ceil(strength * 6))) if sign > 0 else max(4, 12 - max(1, int(np.ceil(strength * 6))))
        mask = np.zeros_like(res, dtype=bool)
        mask[center_h - width // 2:center_h + width // 2, :] = True
        mask[:, center_w - width // 2:center_w + width // 2] = True
        res[mask] = np.clip(res[mask] - 0.05 * strength, MIN_VAL, MAX_VAL)
    return round_clip(res)


def compose_distortion(base, bias_dict):
    total_delta = np.zeros_like(base, dtype=np.float64)
    for idx, bias in bias_dict.items(): total_delta += add_distortion(base, idx, bias) - base
    return round_clip(base + total_delta)


def generate_dataset():
    x_data, y_diag, y_esti = [], [], []
    for param_idx in range(OUT_DIM):
        for _ in range(50):
            diag, esti = np.zeros(OUT_DIM), np.zeros(OUT_DIM)
            bias = np.random.choice([-1, 1]) * np.random.uniform(0.1, MAX_BIAS_ABS)
            diag[param_idx], esti[param_idx] = 1.0, bias
            x_data.append((add_distortion(base_img, param_idx, bias) - base_img)[None, ...]);
            y_diag.append(diag);
            y_esti.append(esti)
    while len(x_data) < N_SAMPLES:
        diag, esti, bias_dict = np.zeros(OUT_DIM), np.zeros(OUT_DIM), {}
        for idx in np.random.choice(OUT_DIM, np.random.randint(2, 5), replace=False):
            bias = np.random.choice([-1, 1]) * np.random.uniform(0.1, MAX_BIAS_ABS)
            diag[idx], esti[idx], bias_dict[idx] = 1.0, bias, bias
        x_data.append((compose_distortion(base_img, bias_dict) - base_img)[None, ...]);
        y_diag.append(diag);
        y_esti.append(esti)
    return np.array(x_data, dtype=np.float32), np.array(y_diag, dtype=np.float32), np.array(y_esti, dtype=np.float32)


class LithoDataset(Dataset):
    def __init__(self, x, d, e): self.x, self.d, self.e = torch.tensor(x), torch.tensor(d), torch.tensor(e)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx): return self.x[idx], self.d[idx], self.e[idx]


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


class EncoderBackbone(nn.Module):
    """基于 ImageNet 预训练 ResNet18 的特征提取器；末端可选 CBAM 以保留注意力可视化。"""
    FEAT_DIM = 512

    def __init__(self, use_cbam=True):
        super().__init__()
        self.use_cbam = use_cbam
        'resnet = models.resnet18(weights=None)'
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # 末端 CBAM：保留 last_sa_map 可视化接口；不影响预训练权重
        self.tail_cbam = CBAM(512) if use_cbam else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 输入若为单通道，扩成 3 通道以匹配预训练 stem
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.tail_cbam(x)
        return self.global_pool(x).flatten(1)

    def freeze_pretrained(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


class ResVGG_DualBranch(nn.Module):
    def __init__(self, use_cbam=True):
        super().__init__()
        self.encoder = EncoderBackbone(use_cbam)
        feat_dim = EncoderBackbone.FEAT_DIM
        self.diag_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(512, OUT_DIM),
                                       nn.Sigmoid())
        self.reg_head = nn.Sequential(nn.Linear(feat_dim + OUT_DIM, 512), nn.ReLU(True), nn.Dropout(0.3),
                                      nn.Linear(512, OUT_DIM))
        self.uncert_head = nn.Sequential(nn.Linear(feat_dim + OUT_DIM, 512), nn.ReLU(True), nn.Linear(512, OUT_DIM))

    def forward(self, x, detach_diag_for_reg=False):
        feat = self.encoder(x)
        diag_prob = self.diag_head(feat)
        reg_signal = diag_prob.detach() if detach_diag_for_reg else diag_prob
        esti_in = torch.cat([feat, reg_signal], dim=1)
        return diag_prob, (torch.tanh(self.reg_head(esti_in)) * MAX_BIAS_ABS) * reg_signal, (
                    F.softplus(self.uncert_head(esti_in)) + 1e-4) * reg_signal + 1e-4


class DirectRegResVGG(nn.Module):
    """消融实验Baseline1：无门控单分支直接回归网络"""

    def __init__(self):
        super().__init__()
        self.encoder = EncoderBackbone(use_cbam=False)
        self.reg_head = nn.Sequential(nn.Linear(EncoderBackbone.FEAT_DIM, 512), nn.ReLU(True), nn.Dropout(0.3),
                                      nn.Linear(512, OUT_DIM))

    def forward(self, x):
        return None, torch.tanh(self.reg_head(self.encoder(x))) * MAX_BIAS_ABS, None


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


def train_model(model, train_loader, val_loader, model_name="Dual_CBAM", is_dual=True):
    # 阶段1：冻结预训练 backbone，只训 head 与末端 CBAM
    if hasattr(model.encoder, "freeze_pretrained"):
        model.encoder.freeze_pretrained()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    bce, mse = nn.BCELoss(), nn.MSELoss()
    unfrozen = False

    for ep in range(EPOCHS):
        # 阶段切换：head 训到一定水平后解冻全网，用更小学习率微调
        if not unfrozen and is_dual and ep >= DYNAMIC_WARMUP_EPOCHS and hasattr(model.encoder, "unfreeze_all"):
            model.encoder.unfreeze_all()
            optimizer = optim.Adam(model.parameters(), lr=LR_PHASE2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - ep, eta_min=MIN_LR)
            unfrozen = True
            print(f"[{model_name}] Epoch {ep + 1}: 解冻 backbone，切换到 PHASE2 微调 (lr={LR_PHASE2})")

        model.train()
        cls_w, reg_w = get_dynamic_loss_weights(ep) if is_dual else (0, 1.0)

        for x, d, e in train_loader:
            x, d, e = x.to(device), d.to(device), e.to(device)
            optimizer.zero_grad()
            if is_dual:
                pd, pe, sigma = model(x)
                loss = cls_w * bce(pd, d) + reg_w * (
                            soft_masked_mae_loss(pe, e, pd.detach()) + UNCERTAINTY_WEIGHT * soft_masked_uncertainty_mae(
                        sigma, pe, e, pd.detach()))
            else:
                _, pe, _ = model(x)
                loss = mse(pe, e)  # 直接回归使用全局MSE
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (ep + 1) % 10 == 0:
            val_acc, val_mae = evaluate_model(model, val_loader, is_dual)
            print(f"[{model_name}] Epoch {ep + 1:4d} | Acc: {val_acc:.4f} | MAE: {val_mae:.6f}")
    return evaluate_model(model, val_loader, is_dual)


def evaluate_model(model, loader, is_dual=True):
    model.eval()
    all_mae, count, accs = 0, 0, []
    with torch.no_grad():
        for x, d, e in loader:
            x, d, e = x.to(device), d.to(device), e.to(device)
            if is_dual:
                pd, pe, _ = model(x)
                accs.append(((pd >= DIAG_THRESHOLD).float() == d).float().mean().item())
                mask = d > DIAG_THRESHOLD
                if mask.sum() > 0:
                    all_mae += torch.abs(pe[mask] - e[mask]).sum().item()
                    count += mask.sum().item()
            else:
                _, pe, _ = model(x)
                accs.append(0.0)  # 单分支无分类精度
                mask = e.abs() > 0
                if mask.sum() > 0:
                    all_mae += torch.abs(pe[mask] - e[mask]).sum().item()
                    count += mask.sum().item()
    return np.mean(accs) if is_dual else 0.0, all_mae / max(count, 1)


# ====================== 【新增核心绘图函数】 ======================

def plot_confusion_matrix(model, loader):
    """绘制诊断分支多标签预测混淆矩阵"""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, d, _ in loader:
            pd, _, _ = model(x.to(device))
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


def plot_attention_map(model, test_img):
    """注意力机制可视化 (模拟物理敏感区域)"""
    model.eval()
    # test_img 已是 (1, 1, H, W)，直接转 tensor 送入，不再 unsqueeze
    img_tensor = torch.tensor(test_img).to(device)
    with torch.no_grad():
        _ = model(img_tensor)  # 触发前向传播
        # 获取 ResNet 末端 CBAM 的空间注意力图 [1, 1, h, w]
        sa_map = model.encoder.tail_cbam.last_sa_map.cpu().numpy()[0, 0]

    # 将小尺寸热力图插值放大到 200x200
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
    print("1. 正在进行高保真光刻数据仿真与增强...")
    x, d, e = generate_dataset()
    m, s = x.mean(), x.std() + 1e-6
    x = (x - m) / s

    xt, xv, dt, dv, et, ev = train_test_split(x, d, e, test_size=0.2, random_state=42)
    train_loader = DataLoader(LithoDataset(xt, dt, et), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LithoDataset(xv, dv, ev), batch_size=BATCH_SIZE)

    metrics = {}

    # === 模型 A: 传统直接回归 Baseline ===
    print("\n2. 训练传统 Baseline 模型 (Direct Regression)...")
    model_base = DirectRegResVGG().to(device)
    acc_base, mae_base = train_model(model_base, train_loader, val_loader, "Base", is_dual=False)
    metrics['Base'] = (0.0, mae_base)  # 单分支无诊断

    # === 模型 B: 双分支 (无注意力) ===
    print("\n3. 训练双分支消融模型 (Dual-Branch without CBAM)...")
    model_dual_no_cbam = ResVGG_DualBranch(use_cbam=False).to(device)
    acc_d1, mae_d1 = train_model(model_dual_no_cbam, train_loader, val_loader, "Dual_NoCBAM", is_dual=True)
    metrics['Dual_NoCBAM'] = (acc_d1, mae_d1)

    # === 模型 C: 提出模型 (双分支 + CBAM) ===
    print("\n4. 训练提出模型 (Proposed Dual-Branch + CBAM)...")
    model_dual_cbam = ResVGG_DualBranch(use_cbam=True).to(device)
    acc_d2, mae_d2 = train_model(model_dual_cbam, train_loader, val_loader, "Dual_CBAM", is_dual=True)
    metrics['Dual_CBAM'] = (acc_d2, mae_d2)




    print("\n================ 综合性能测试报告 ================")
    print(f"Baseline 直接回归  | MAE: {mae_base:.5f}")
    print(f"双分支无注意力     | Acc: {acc_d1:.4f}, MAE: {mae_d1:.5f}")
    print(f"提出模型(完整态)   | Acc: {acc_d2:.4f}, MAE: {mae_d2:.5f}")
    print("==================================================")

    # 生成论文图表
    print("\n5. 正在生成科研图表...")
    plot_confusion_matrix(model_dual_cbam, val_loader)
    plot_ablation_study(metrics)

    # 取一张有特征扰动的差异图测试注意力机制
    test_img = xv[0:1]
    plot_attention_map(model_dual_cbam, test_img)
    print("全部流程执行完毕！")


if __name__ == "__main__":
    main()