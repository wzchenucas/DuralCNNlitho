import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================================================
# 1. 核心控制台 (配置你的参数与输出数量)
# =========================================================
data_dir = r"D:\pycharm\DuralNetwork\data"

IDEAL_PARAMS = {
    'X': 10,
    'Y': 10,
    'Height': 10,
    'LS': 0,
    'Sidewall Angle': 90
}

# 【新增】三权分立：独立控制三种输出的数量！
MAX_PRINT_LINES = 100  # 1. 控制台：最多打印多少行纯文本日志
MAX_PLOT_IMAGES = 10  # 2. 画板图：最多渲染多少张灰度矩阵图 (建议较小，防卡顿)
MAX_TABLE_ROWS = 10  # 3. 诊断表：醒目变色表格里最多显示多少行数据 (可以设大一点)

TARGET_PARAMS = list(IDEAL_PARAMS.keys())

# =========================================================
# 2. 动态扫描文件与解析
# =========================================================
dat_files = glob.glob(os.path.join(data_dir, '*.dat'))
if not dat_files:
    raise FileNotFoundError(f"在目录 {data_dir} 下未找到任何 .dat 文件。")

print(f">> 扫描完毕：共找到 {len(dat_files)} 个 .dat 文件。开始批量解析...")

all_images = []
all_labels_dict = []
all_raw_headers = []
all_source_files = []

for file_path in dat_files:
    file_name = os.path.basename(file_path)

    # === 【修改 1】根据文件名开头是否为 LS (忽略大小写) 来设定 LS 属性 ===
    is_ls_pattern = 1.0 if file_name.upper().startswith('LS') else 0.0

    current_matrix = []
    # === 【修改 2】初始化字典时，直接将 LS 属性强制写入 ===
    current_label = {'LS': is_ls_pattern}
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
                all_images.append(current_matrix)
                all_labels_dict.append(current_label.copy())
                all_raw_headers.append(current_header)
                all_source_files.append(file_name)

                current_matrix = []
                # === 【修改 3】在一个文件包含多个矩阵重置字典时，重新保留 LS 属性 ===
                current_label = {'LS': is_ls_pattern}
                current_header = ""

            for param in TARGET_PARAMS:
                pattern = rf'\b{param}\b.*?=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
                m = re.search(pattern, line, re.IGNORECASE)
                if m: current_label[param] = float(m.group(1))

            current_header += line + " | "

    if current_matrix:
        all_images.append(current_matrix)
        all_labels_dict.append(current_label.copy())
        all_raw_headers.append(current_header)
        all_source_files.append(file_name)

num_images = len(all_images)
print(f">> 解析完成！提取出 {num_images} 组数据。")

# =========================================================
# 3. 动态生成双分支标签并打包
# =========================================================
all_images_np = [np.array(mat) for mat in all_images]
shapes = [mat.shape for mat in all_images_np]
unique_shapes = set(shapes)

if len(unique_shapes) == 1:
    X_data = np.array(all_images_np)
else:
    X_data = np.empty(num_images, dtype=object)
    X_data[:] = all_images_np

Y_reg_data = np.array([[d.get(p, IDEAL_PARAMS[p]) for p in TARGET_PARAMS] for d in all_labels_dict])
Y_cls_data = np.array([
    [1 if abs(d.get(p, IDEAL_PARAMS[p]) - IDEAL_PARAMS[p]) > 1e-5 else 0 for p in TARGET_PARAMS]
    for d in all_labels_dict
])

output_file = os.path.join(data_dir, 'uda_dataset.npz')
np.savez(output_file, images=X_data, labels_reg=Y_reg_data, labels_cls=Y_cls_data, headers=all_raw_headers,
         source_files=all_source_files)

print(f"\n>> 数据已保存至: {output_file}")
print(f">> 特征 X 形状: {X_data.shape}")
print(f">> 回归标签 Y_reg: {Y_reg_data.shape} | 分类标签 Y_cls: {Y_cls_data.shape}")

# =========================================================
# 4. 控制台 Print 输出核对
# =========================================================
print_count = min(num_images, MAX_PRINT_LINES)
plot_count = min(num_images, MAX_PLOT_IMAGES)
table_count = min(num_images, MAX_TABLE_ROWS)

print(f"\n>> 正在为您抽查前 {print_count} 组数据的参数与偏差状态：")
print("-" * 90)
for idx in range(print_count):
    label_dict = all_labels_dict[idx]
    img_shape = all_images_np[idx].shape
    res_str = f"[{img_shape[0]}x{img_shape[1]}]"
    file_str = f"[{all_source_files[idx]}]"

    param_strs = []
    for p in TARGET_PARAMS:
        val = label_dict.get(p, IDEAL_PARAMS[p])
        cls_flag = 1 if abs(val - IDEAL_PARAMS[p]) > 1e-5 else 0
        is_padded = "(补)" if p not in label_dict else ""
        param_strs.append(f"{p}={val}{is_padded}({cls_flag})")

    param_str = ", ".join(param_strs)
    print(f"Image {idx + 1:02d} {file_str} {res_str} --> {param_str}")
print("-" * 90 + "\n")

# =========================================================
# 5. 可视化图表生成 (图1：灰度矩阵，图2：分类醒目表格)
# =========================================================

if plot_count > 0:
    cols = int(np.ceil(np.sqrt(plot_count)))
    rows = int(np.ceil(plot_count / cols))
    fig_img, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig_img.canvas.manager.set_window_title(f'Images View (Showing {plot_count} items)')

    if plot_count == 1: axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(plot_count):
        ax = axes[idx]
        im = ax.imshow(all_images_np[idx], cmap='jet')
        ax.set_xticks([]);
        ax.set_yticks([])

        params_list = []
        for p in TARGET_PARAMS:
            val = all_labels_dict[idx].get(p, IDEAL_PARAMS[p])
            cls_flag = 1 if abs(val - IDEAL_PARAMS[p]) > 1e-5 else 0
            params_list.append(f"{p}:{val}({cls_flag})")

        title_str = "\n".join([", ".join(params_list[i:i + 2]) for i in range(0, len(params_list), 2)])
        ax.set_title(title_str, fontsize=9, pad=6)

    for idx in range(plot_count, len(axes)):
        fig_img.delaxes(axes[idx])
    fig_img.tight_layout()

if table_count > 0:
    fig_height = max(5, 0.45 * table_count + 1)
    fig_tab, ax_tab = plt.subplots(figsize=(10, fig_height))
    fig_tab.canvas.manager.set_window_title(f'Classification Labels Table (Showing {table_count} items)')
    ax_tab.axis('off')

    table_cols = ['Source File'] + TARGET_PARAMS
    cell_text = []
    cell_colors = []

    for idx in range(table_count):
        row_text = [all_source_files[idx]]
        row_colors = ['#f0f0f0']

        for p in TARGET_PARAMS:
            val = all_labels_dict[idx].get(p, IDEAL_PARAMS[p])
            cls_flag = 1 if abs(val - IDEAL_PARAMS[p]) > 1e-5 else 0

            cell_str = f"{p}\n{val}\n({cls_flag})"
            row_text.append(cell_str)

            color = '#ff9999' if cls_flag == 1 else '#ffffff'
            row_colors.append(color)

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax_tab.table(cellText=cell_text,
                         cellColours=cell_colors,
                         colLabels=table_cols,
                         loc='center',
                         cellLoc='center')

    table.scale(1, 3.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9d9d9')

    fig_tab.tight_layout()

if plot_count > 0 or table_count > 0:
    plt.show()