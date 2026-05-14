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
OUT_DIM = 6          # 【修改】[光板X, 光板Y, 光板H, 有图X, 有图Y, 有图H]
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

DATA_PATH = "data/uda_dataset.npz"
PARAM_NAMES = ["LS0_X", "LS0_Y", "LS0_H", "LS1_X", "LS1_Y", "LS1_H"]

# ====================== 数据加载 ======================
def load_dataset():
    data = np.load(DATA_PATH)
    images  = data['images'].astype(np.float32)       
    y_reg = data['labels_reg'].astype(np.float32)   # (N, 6)
    y_cls = data['labels_cls'].astype(np.int64)     # (N, 6)

    # 提取有效数据的掩码，用于忽略无图形区域的 0.0 占位符
    # 数据本身X/Y/H不可能为0，因此值为0即代表占位符
    domain_mask = (y_reg != 0.0).astype(np.float32)

    reg_min = np.zeros(OUT_DIM, dtype=np.float32)
    reg_max = np.ones(OUT_DIM,  dtype=np.float32)
    
    # 仅在有物理意义的激活域内进行归一化极值计算
    for i in range(OUT_DIM):
        valid_vals = y_reg[domain_mask[:, i] == 1, i]
        if len(valid_vals) > 0:
            reg_min[i] = valid_vals.min()
            reg_max[i] = valid_vals.max()

    denom = (reg_max - reg_min).clip(min=1e-6)
    y_reg_norm = np.zeros_like(y_reg)
    for i in range(OUT_DIM):
        valid_idx = (domain_mask[:, i] == 1)
        y_reg_norm[valid_idx, i] = (y_reg[valid_idx, i] - reg_min[i]) / denom[i]

    x = images[:, np.newaxis, :, :]
    m, s = x.mean(), x.std() + 1e-6
    x = (x - m) / s

    # 移除 ls_onehot 返回项，新增 domain_mask 以精准屏蔽非激活损失
    return x, y_cls.astype(np.float32), y_reg_norm, domain_mask, reg_min, reg_max


class LithoDataset(Dataset):
    def __init__(self, x, d, e, m):
        self.x = torch.tensor(x)
        self.d = torch.tensor(d)
        self.e = torch.tensor(e)
        self.m = torch.tensor(m)  # Domain mask (N, 6)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.d[idx], self.e[idx], self.m[idx]

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
        self.last_sa_map = None

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
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False), nn.BatchNorm2d(planes * 4))
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
        self.res1 = ResNetBottleneck(64, 64, 1, use_cbam=False)
        self.res2 = ResNetBottleneck(256, 128, 2, use_cbam=False)
        self.res3 = ResNetBottleneck(512, 256, 2, use_cbam=use_cbam)
        self.res4 = ResNetBottleneck(1024, 512, 2, use_cbam=use_cbam)
        
        self.vgg_block1 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.vgg_cbam1 = CBAM(128) if use_cbam else nn.Identity()
        self.vgg_block2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.vgg_cbam2 = CBAM(256) if use_cbam else nn.Identity()
        self.vgg_block3 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True))
        
        self.fusion = nn.Sequential(nn.Linear(2048 + 512, 2560, bias=False), nn.BatchNorm1d(2560), nn.ReLU(inplace=True))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    # 【修改】去掉 ls_onehot，仅依靠6维解耦自我识别
    def forward(self, x):
        stem_out = self.stem(x)
        res_out = self.global_pool(self.res4(self.res3(self.res2(self.res1(stem_out))))).flatten(1)  
        vgg_out = self.global_pool(self.vgg_block3(self.vgg_cbam2(self.vgg_block2(self.vgg_cbam1(self.vgg_block1(stem_out)))))).flatten(1) 
        return self.fusion(torch.cat([res_out, vgg_out], dim=1))   

class ResVGG_DualBranch(nn.Module):
    def __init__(self, use_cbam=True):
        super().__init__()
        self.encoder = EncoderBackbone(use_cbam)
        feat_dim = 2560  
        self.diag_head = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(512, OUT_DIM))   
        self.reg_head = nn.Sequential(nn.Linear(feat_dim + OUT_DIM, 512), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(512, OUT_DIM))
        self.uncert_head = nn.Sequential(nn.Linear(feat_dim + OUT_DIM, 512), nn.ReLU(True), nn.Linear(512, OUT_DIM))

    def forward(self, x, detach_diag_for_reg=False):
        feat = self.encoder(x)                                   
        diag_logits = self.diag_head(feat)                                  
        diag_prob   = torch.sigmoid(diag_logits)                            
        reg_signal  = diag_prob.detach() if detach_diag_for_reg else diag_prob
        hard_gate   = (diag_prob.detach() >= DIAG_THRESHOLD).float()
        gate        = hard_gate if not self.training else reg_signal        
        esti_in     = torch.cat([feat, reg_signal], dim=1)
        return diag_logits, torch.sigmoid(self.reg_head(esti_in)) * gate, \
               (F.softplus(self.uncert_head(esti_in)) + 1e-4) * gate + 1e-4

class DirectRegResVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderBackbone(use_cbam=False)
        self.reg_head = nn.Sequential(nn.Linear(2560, 512), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(512, OUT_DIM))

    def forward(self, x):
        return None, torch.sigmoid(self.reg_head(self.encoder(x))), None

# ====================== 损失与训练逻辑 ======================
def soft_masked_mae_loss(pred, target, soft_mask): 
    return (torch.abs(pred - target) * soft_mask.clamp(0, 1)).sum() / soft_mask.sum().clamp_min(1e-6)

def soft_masked_uncertainty_mae(pred_sigma, pred_mean, target, soft_mask): 
    return (torch.abs(pred_sigma - torch.abs(pred_mean.detach() - target)) * soft_mask.clamp(0, 1)).sum() / soft_mask.sum().clamp_min(1e-6)

def get_dynamic_loss_weights(epoch):
    if epoch < DYNAMIC_WARMUP_EPOCHS: return DYNAMIC_WEIGHT_INIT_CLS, 1.0 - DYNAMIC_WEIGHT_INIT_CLS
    ratio = (epoch - DYNAMIC_WARMUP_EPOCHS) / (EPOCHS - DYNAMIC_WARMUP_EPOCHS)
    cls_w = max(DYNAMIC_WEIGHT_FINAL_CLS, DYNAMIC_WEIGHT_INIT_CLS - ratio * (DYNAMIC_WEIGHT_INIT_CLS - DYNAMIC_WEIGHT_FINAL_CLS))
    return cls_w, 1.0 - cls_w

def train_model(model, train_loader, val_loader, reg_min, reg_max, model_name="Dual_CBAM", is_dual=True):
    optimizer = optim.Adam(model.parameters(), lr=LR_PHASE1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    
    # 【新增】基于实际数据集自动动态计算列级别的 pos_weight，防范样本稀缺崩溃
    ds_cls = train_loader.dataset.d
    pos_w_list = []
    for i in range(OUT_DIM):
        pos_cnt = (ds_cls[:, i] == 1).sum().item()
        neg_cnt = (ds_cls[:, i] == 0).sum().item()
        pos_w_list.append(neg_cnt / max(pos_cnt, 1.0))
    
    bce_logits = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w_list, device=device))

    for ep in range(EPOCHS):
        model.train()
        cls_w, reg_w = get_dynamic_loss_weights(ep) if is_dual else (0, 1.0)

        for x, d, e, m in train_loader:
            x, d, e, m = x.to(device), d.to(device), e.to(device), m.to(device)
            optimizer.zero_grad()
            if is_dual:
                diag_logits, pe, sigma = model(x)
                diag_prob = torch.sigmoid(diag_logits.detach())
                
                # 双重保护：仅在有物理意义的数据域上施加MAE惩罚
                eff_mask = diag_prob * m 
                loss = cls_w * bce_logits(diag_logits, d) + reg_w * (
                    soft_masked_mae_loss(pe, e, eff_mask) +
                    UNCERTAINTY_WEIGHT * soft_masked_uncertainty_mae(sigma, pe, e, eff_mask))
            else:
                _, pe, _ = model(x)
                # 直接回归通过 m 阻止未激活域的梯度流出
                mse_unreduced = F.mse_loss(pe, e, reduction='none')
                loss = (mse_unreduced * m).sum() / m.sum().clamp_min(1.0)
                
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (ep + 1) % 10 == 0:
            val_acc, val_mae = evaluate_model(model, val_loader, reg_min, reg_max, is_dual)
            print(f"[{model_name}] Epoch {ep + 1:4d} | Acc: {val_acc:.4f} | MAE: {val_mae:.4f}")
    return evaluate_model(model, val_loader, reg_min, reg_max, is_dual)

def evaluate_model(model, loader, reg_min, reg_max, is_dual=True):
    model.eval()
    t_min = torch.tensor(reg_min, device=device)                         
    t_rng = torch.tensor(reg_max - reg_min, device=device).clamp(1e-6)  

    col_mae = torch.zeros(OUT_DIM, device=device)  
    col_count = torch.zeros(OUT_DIM, device=device)  
    accs = []

    with torch.no_grad():
        for x, d, e, m in loader:
            x, d, e, m = x.to(device), d.to(device), e.to(device), m.to(device)
            if is_dual:
                diag_logits, pe, _ = model(x)
                diag_prob = torch.sigmoid(diag_logits)
                pred_cls  = (diag_prob >= DIAG_THRESHOLD).float()
                
                # 在分类计算中，依靠域掩码 m 将 0-占位符位进行屏蔽（占位符准确率是100%没意义的）
                valid_preds = (pred_cls == d).float() * m
                accs.append(valid_preds.sum().item() / m.sum().item() if m.sum() > 0 else 0)
                
                # 回归 MAE 依然只统计实际有偏差的地方 (且属于激活域)
                col_mask = (d > DIAG_THRESHOLD) * m                          
            else:
                _, pe, _ = model(x)
                accs.append(0.0)
                col_mask = (e > 0) * m                                       

            pe_phys = pe * t_rng + t_min   
            e_phys  = e  * t_rng + t_min   
            abs_err = torch.abs(pe_phys - e_phys)                        
            col_mae   += (abs_err * col_mask.float()).sum(dim=0)
            col_count += col_mask.float().sum(dim=0)

    valid_cols = col_count > 0
    per_col_mae = torch.where(valid_cols, col_mae / col_count.clamp(1), torch.zeros_like(col_mae))
    mean_mae = per_col_mae[valid_cols].mean().item() if valid_cols.any() else 0.0
    
    return np.mean(accs) if len(accs)>0 and is_dual else 0.0, mean_mae

# ====================== 核心绘图函数 ======================
def plot_confusion_matrix(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, d, _, m in loader:
            diag_logits, _, _ = model(x.to(device))
            pd = torch.sigmoid(diag_logits)
            
            # 只抽取当前激活域内的真实标签来进行混淆矩阵统计
            mask_cpu = m.cpu().numpy() == 1
            all_true.extend(d.cpu().numpy()[mask_cpu])
            all_pred.extend((pd.cpu().numpy()[mask_cpu] > DIAG_THRESHOLD).astype(int))

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Normal", "Defected"], yticklabels=["Normal", "Defected"])
    plt.title("Diagnosis Branch Confusion Matrix\n(EUV Multi-parameter Decoupling)")
    plt.ylabel('True Status'); plt.xlabel('Predicted Status')
    plt.tight_layout()
    plt.savefig('fig_confusion_matrix.png', dpi=300)
    plt.close()

def plot_ablation_study(metrics_dict):
    labels = ['Baseline\n(Direct Reg)', 'Dual-Branch\n(No CBAM)', 'Proposed\n(Dual + CBAM)']
    acc_data = [metrics_dict['Base'][0], metrics_dict['Dual_NoCBAM'][0], metrics_dict['Dual_CBAM'][0]]
    mae_data = [metrics_dict['Base'][1], metrics_dict['Dual_NoCBAM'][1], metrics_dict['Dual_CBAM'][1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(x - width / 2, acc_data, width, label='Diagnosis Acc', color='#4C72B0')
    ax1.set_ylabel('Accuracy', color='#4C72B0', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, mae_data, width, label='Estimation MAE', color='#C44E52')
    ax2.set_ylabel('Mean Absolute Error (nm)', color='#C44E52', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    plt.title('Ablation Study on Separated Light-Board/Pattern Layout', fontsize=14)

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels1 + labels2, loc='center right')
    plt.tight_layout()
    plt.savefig('fig_ablation_study.png', dpi=300)
    plt.close()

def plot_attention_map(model, test_img):
    model.eval()
    img_tensor = torch.tensor(test_img).to(device)
    with torch.no_grad():
        _ = model(img_tensor)
        sa_map = model.encoder.res4.cbam.last_sa_map.cpu().numpy()[0, 0]

    sa_map_resized = torch.nn.functional.interpolate(
        torch.from_numpy(sa_map).float().unsqueeze(0).unsqueeze(0),
        size=(IMG_H, IMG_W), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    display_img = test_img[0, 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_img, cmap='gray'); axes[0].set_title("Input EUV Difference"); axes[0].axis('off')
    axes[1].imshow(sa_map_resized, cmap='jet'); axes[1].set_title("CBAM Spatial Attention Heatmap"); axes[1].axis('off')
    axes[2].imshow(display_img, cmap='gray'); axes[2].imshow(sa_map_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay (Focus on Defect Physics)"); axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('fig_attention_visualization.png', dpi=300)
    plt.close()

def main():
    print("1. 加载真实 EUV 仿真数据集...")
    x, d, e, m, reg_min, reg_max = load_dataset()
    print(f"   数据量: {len(x)}, 图像尺寸: {x.shape[2]}x{x.shape[3]}, 任务维度: {OUT_DIM}维解耦")
    print(f"   回归极值: {dict(zip(PARAM_NAMES, [f'{reg_min[i]:.2f}~{reg_max[i]:.2f}' for i in range(OUT_DIM)]))}")

    xt, xv, dt, dv, et, ev, mt, mv = train_test_split(x, d, e, m, test_size=0.2, random_state=42)
    train_loader = DataLoader(LithoDataset(xt, dt, et, mt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(LithoDataset(xv, dv, ev, mv), batch_size=BATCH_SIZE)

    metrics = {}
    print("\n2. 训练传统 Baseline 模型 (Direct Regression)...")
    model_base = DirectRegResVGG().to(device)
    acc_base, mae_base = train_model(model_base, train_loader, val_loader, reg_min, reg_max, "Base", is_dual=False)
    metrics['Base'] = (0.0, mae_base)

    print("\n3. 训练双分支消融模型 (Dual-Branch without CBAM)...")
    model_dual_no_cbam = ResVGG_DualBranch(use_cbam=False).to(device)
    acc_d1, mae_d1 = train_model(model_dual_no_cbam, train_loader, val_loader, reg_min, reg_max, "Dual_NoCBAM", is_dual=True)
    metrics['Dual_NoCBAM'] = (acc_d1, mae_d1)

    print("\n4. 训练提出模型 (Proposed Dual-Branch + CBAM)...")
    model_dual_cbam = ResVGG_DualBranch(use_cbam=True).to(device)
    acc_d2, mae_d2 = train_model(model_dual_cbam, train_loader, val_loader, reg_min, reg_max, "Dual_CBAM", is_dual=True)
    metrics['Dual_CBAM'] = (acc_d2, mae_d2)

    print("\n================ 综合性能测试报告 ================")
    print(f"Baseline 直接回归  | MAE: {mae_base:.4f} nm")
    print(f"双分支无注意力     | Acc: {acc_d1:.4f}, MAE: {mae_d1:.4f} nm")
    print(f"提出模型(完整态)   | Acc: {acc_d2:.4f}, MAE: {mae_d2:.4f} nm")
    print("==================================================")

    print("\n5. 正在生成科研图表...")
    plot_confusion_matrix(model_dual_cbam, val_loader)
    plot_ablation_study(metrics)

    test_img = xv[0:1]
    plot_attention_map(model_dual_cbam, test_img)
    print("全部流程执行完毕！")

if __name__ == "__main__":
    main()
