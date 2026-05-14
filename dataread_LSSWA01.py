import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================================================
# 1. 核心控制台 (配置你的参数与输出数量)
# =========================================================
data_dir = r"D:\pycharm\DuralNetwork\data"

# 【修改】暂时将SWA注释或固定，目标参数改为 X, Y, Height
IDEAL_PARAMS = {
    'X': 10.0,
    'Y': 10.0,
    'Height': 10.0,
    # 'Sidewall Angle': 90  # 等后续数据加进来再取消注释
}

MAX_PRINT_LINES = 100
MAX_PLOT_IMAGES = 10 
MAX_TABLE_ROWS = 10  

TARGET_PARAMS = ['X', 'Y', 'Height']
# 6维输出顺序解耦
PARAM_NAMES_6D = ['LS0_X', 'LS0_Y', 'LS0_H', 'LS1_X', 'LS1_Y', 'LS1_H']

# =========================================================
# 2. 动态扫描文件与解析及【去重防泄漏逻辑】
# =========================================================
dat_files = glob.glob(os.path.join(data_dir, '*.dat'))
if not dat_files:
    raise FileNotFoundError(f"在目录 {data_dir} 下未找到任何 .dat 文件。")

print(f">> 扫描完毕：共找到 {len(dat_files)} 个 .dat 文件。开始批量解析...")

all_images = []
all_labels_reg = []
all_labels_cls = []
all_raw_headers = []
all_source_files = []

# 【新增】哈希记录集合，用于第一张图的去重防泄漏
seen_base_images = set()

for file_path in dat_files:
    file_name = os.path.basename(file_path)
    is_ls_pattern = 1 if file_name.upper().startswith('LS') else 0

    file_matrices = []
    file_labels = []
    file_headers = []

    current_matrix = []
    current_label = {}
    current_header = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue

        parts = line.split()
        try:
            num_row = [float(x) for x in parts]
            current_matrix.append(num_row)
        except ValueError:
            if current_matrix:
                file_matrices.append(current_matrix)
                file_labels.append(current_label.copy())
                file_headers.append(current_header)
                current_matrix = []
                current_label = {}
                current_header = ""

            for param in TARGET_PARAMS:
                pattern = rf'\b{param}\b.*?=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
                m = re.search(pattern, line, re.IGNORECASE)
                if m: current_label[param] = float(m.group(1))

            current_header += line + " | "

    if current_matrix:
        file_matrices.append(current_matrix)
        file_labels.append(current_label.copy())
        file_headers.append(current_header)
        
    if not file_matrices: continue

    # ================= 去重与 6 维解耦标签生成 =================
    first_img_np = np.array(file_matrices[0], dtype=np.float32)
    img_hash = hash(first_img_np.tobytes())
    key = (is_ls_pattern, img_hash)

    for i in range(len(file_matrices)):
        # 检查本组的第一张图是否在其他组出现过
        if i == 0:
            if key in seen_base_images:
                continue # 如果存在，则丢弃该重复基准图
            else:
                seen_base_images.add(key)
        
        label_dict = file_labels[i]
        val_x = label_dict.get('X', IDEAL_PARAMS['X'])
        val_y = label_dict.get('Y', IDEAL_PARAMS['Y'])
        val_h = label_dict.get('Height', IDEAL_PARAMS['Height'])

        cls_x = 1 if abs(val_x - IDEAL_PARAMS['X']) > 1e-5 else 0
        cls_y = 1 if abs(val_y - IDEAL_PARAMS['Y']) > 1e-5 else 0
        cls_h = 1 if abs(val_h - IDEAL_PARAMS['Height']) > 1e-5 else 0

        # 【核心修改】：通过 LS 域将特征解耦到前 3 维或后 3 维，另外一半用 0 占位
        if is_ls_pattern == 0: # 光板
            reg_6d = [val_x, val_y, val_h, 0.0, 0.0, 0.0]
            cls_6d = [cls_x, cls_y, cls_h, 0, 0, 0]
        else: # 有图形
            reg_6d = [0.0, 0.0, 0.0, val_x, val_y, val_h]
            cls_6d = [0, 0, 0, cls_x, cls_y, cls_h]

        all_images.append(file_matrices[i])
        all_labels_reg.append(reg_6d)
        all_labels_cls.append(cls_6d)
        all_raw_headers.append(file_headers[i])
        all_source_files.append(file_name)

num_images = len(all_images)
print(f">> 解析完成！(含首图去重) 提取出 {num_images} 组有效数据。")

# =========================================================
# 3. 数据打包
# =========================================================
all_images_np = [np.array(mat) for mat in all_images]
shapes = [mat.shape for mat in all_images_np]
if len(set(shapes)) == 1:
    X_data = np.array(all_images_np)
else:
    X_data = np.empty(num_images, dtype=object)
    X_data[:] = all_images_np

Y_reg_data = np.array(all_labels_reg, dtype=np.float32)
Y_cls_data = np.array(all_labels_cls, dtype=np.int64)

output_file = os.path.join(data_dir, 'uda_dataset.npz')
np.savez(output_file, images=X_data, labels_reg=Y_reg_data, labels_cls=Y_cls_data, headers=all_raw_headers, source_files=all_source_files)

print(f"\n>> 数据已保存至: {output_file}")
print(f">> 特征 X 形状: {X_data.shape}")
print(f">> 回归标签 Y_reg: {Y_reg_data.shape} (6维) | 分类标签 Y_cls: {Y_cls_data.shape} (6维)")
