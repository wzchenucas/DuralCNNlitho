import os
import glob
import re
import numpy as np

# =========================================================
# 配置
# =========================================================
DATA_DIR   = r"D:\pycharm\DuralNetwork\data"
OUTPUT_NPZ = os.path.join(DATA_DIR, "uda_dataset.npz")

# 基准值：参数等于此值时视为"正常"(cls=0)
IDEAL_PARAMS = {'X': 10.0, 'Y': 10.0, 'Height': 10.0}
TARGET_PARAMS = ['X', 'Y', 'Height']   # 输出顺序固定

# =========================================================
# 1. 从单行 header 文本中提取参数值
#    格式示例：
#      .../FWHM/X = 10.5 nm
#      .../FWHM/Y = 10.5 nm
#      .../Bottom/Height = 10.5 nm
# =========================================================
_PARAM_PATTERNS = {
    'X':      re.compile(r'FWHM/X\s*=\s*([\d.]+)',      re.IGNORECASE),
    'Y':      re.compile(r'FWHM/Y\s*=\s*([\d.]+)',      re.IGNORECASE),
    'Height': re.compile(r'/Height\s*=\s*([\d.]+)',      re.IGNORECASE),
}

def extract_params_from_header(header_text):
    """从合并后的 header 字符串中提取各参数值，未出现则返回基准值。"""
    result = {}
    for key, pat in _PARAM_PATTERNS.items():
        m = pat.search(header_text)
        result[key] = float(m.group(1)) if m else IDEAL_PARAMS[key]
    return result


# =========================================================
# 2. 解析单个 .dat 文件 → 返回 (matrices, param_list, header_list)
#    每个 Analysis Node 对应一个矩阵块和一组参数
# =========================================================
def parse_dat_file(filepath):
    is_ls = os.path.basename(filepath).upper().startswith('LS')

    matrices, param_list, header_list = [], [], []
    cur_mat, cur_header_lines = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cur_mat.append([float(x) for x in parts])
            except ValueError:
                # 遇到非数字行 → 若之前有矩阵数据则先存档
                if cur_mat:
                    arr = np.array(cur_mat, dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[0] > 1:
                        header_text = ' | '.join(cur_header_lines)
                        params = extract_params_from_header(header_text)
                        params['LS'] = float(is_ls)
                        matrices.append(arr)
                        param_list.append(params)
                        header_list.append(header_text)
                    cur_mat = []
                    cur_header_lines = []
                cur_header_lines.append(s)

    # 文件末尾最后一个矩阵块
    if cur_mat:
        arr = np.array(cur_mat, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > 1:
            header_text = ' | '.join(cur_header_lines)
            params = extract_params_from_header(header_text)
            params['LS'] = float(is_ls)
            matrices.append(arr)
            param_list.append(params)
            header_list.append(header_text)

    return matrices, param_list, header_list, is_ls


# =========================================================
# 3. 跨文件去重：同一大类 (LS/光板) 内若首帧图像完全相同，只保留一份
#    策略：以第一帧的字节指纹 (tobytes hash) 为 key，同类内去重
# =========================================================
def deduplicate_within_class(all_data):
    """
    all_data: list of (image_np, params_dict, header_str, source_file, is_ls)
    返回去重后的同结构列表，并打印去除数量。
    """
    seen = {}   # (is_ls, img_hash) -> True
    kept, dropped = [], 0

    for item in all_data:
        img, params, header, src, is_ls = item
        img_hash = hash(img.tobytes())
        key = (is_ls, img_hash)
        if key in seen:
            dropped += 1
        else:
            seen[key] = True
            kept.append(item)

    print(f"  去重：共移除 {dropped} 个重复帧，保留 {len(kept)} 帧")
    return kept


# =========================================================
# 4. 主流程
# =========================================================
def main():
    dat_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.dat')))
    if not dat_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下未找到任何 .dat 文件")

    print(f"找到 {len(dat_files)} 个 .dat 文件，开始解析...")

    # --- 4.1 解析所有文件 ---
    raw_data = []   # (image, params, header, source_file, is_ls)
    for fpath in dat_files:
        fname = os.path.basename(fpath)
        mats, params_list, headers, is_ls = parse_dat_file(fpath)
        print(f"  {fname}: 解析到 {len(mats)} 帧, LS={int(is_ls)}")
        for img, params, hdr in zip(mats, params_list, headers):
            raw_data.append((img, params, hdr, fname, is_ls))

    print(f"解析完毕，共 {len(raw_data)} 帧")

    # --- 4.2 同大类内去重（移除首帧相同的重复帧）---
    print("执行帧去重...")
    raw_data = deduplicate_within_class(raw_data)

    # --- 4.3 构建 numpy 数组 ---
    images_list  = [item[0] for item in raw_data]
    params_list  = [item[1] for item in raw_data]
    headers_list = [item[2] for item in raw_data]
    sources_list = [item[3] for item in raw_data]

    # 统一形状检查
    shapes = set(img.shape for img in images_list)
    if len(shapes) > 1:
        raise ValueError(f"图像尺寸不一致: {shapes}")

    X_data = np.stack(images_list)   # (N, H, W)

    # 回归标签：[X, Y, Height, LS]，顺序与 TARGET_PARAMS + LS 对应
    FULL_PARAMS = TARGET_PARAMS + ['LS']
    Y_reg = np.array([[p.get(k, IDEAL_PARAMS.get(k, 0.0)) for k in FULL_PARAMS]
                      for p in params_list], dtype=np.float32)

    # 分类标签：参数偏离基准 > 1e-5 则为 1，LS 列直接取值
    Y_cls = np.zeros_like(Y_reg, dtype=np.int64)
    for i, p in enumerate(params_list):
        for j, key in enumerate(FULL_PARAMS):
            val = p.get(key, IDEAL_PARAMS.get(key, 0.0))
            if key == 'LS':
                Y_cls[i, j] = int(val)
            else:
                Y_cls[i, j] = 1 if abs(val - IDEAL_PARAMS[key]) > 1e-5 else 0

    # --- 4.4 保存 ---
    np.savez(
        OUTPUT_NPZ,
        images=X_data,
        labels_reg=Y_reg,
        labels_cls=Y_cls,
        headers=np.array(headers_list),
        source_files=np.array(sources_list),
    )

    print(f"\n已保存至: {OUTPUT_NPZ}")
    print(f"  images    : {X_data.shape}  dtype={X_data.dtype}")
    print(f"  labels_reg: {Y_reg.shape}   列顺序={FULL_PARAMS}")
    print(f"  labels_cls: {Y_cls.shape}")

    # --- 4.5 简要核对 ---
    print("\n前10条样本核对:")
    print(f"{'#':<4} {'文件':<30} {'X':>6} {'Y':>6} {'H':>6} {'LS':>4} "
          f"{'cls_X':>6} {'cls_Y':>6} {'cls_H':>6}")
    print("-" * 80)
    for i in range(min(10, len(raw_data))):
        r = Y_reg[i]; c = Y_cls[i]; src = sources_list[i]
        print(f"{i:<4} {src:<30} {r[0]:>6.1f} {r[1]:>6.1f} {r[2]:>6.1f} {r[3]:>4.0f} "
              f"{c[0]:>6} {c[1]:>6} {c[2]:>6}")

    # 统计各列分布
    print("\n标签统计:")
    for j, key in enumerate(FULL_PARAMS):
        vals = Y_reg[:, j]
        cls_ones = Y_cls[:, j].sum()
        print(f"  {key}: reg=[{vals.min():.1f}, {vals.max():.1f}]  cls=1: {cls_ones}/{len(vals)}")


if __name__ == "__main__":
    main()
