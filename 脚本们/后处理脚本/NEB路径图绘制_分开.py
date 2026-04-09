"""
NEB路径图绘制脚本 - 分别绘制
将每条路径单独绘制保存，保持格式一致
需要安装 scipy: pip install scipy
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, PchipInterpolator
import os

# ========== 字体与画板设置 (保持一致) ==========
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 2  # 边框线宽

# ========== 数据输入区域 ==========
# 格式：{"label": "图例名称", "path": [X轴数据], "energy": [Y轴数据], "color": "颜色"}
# Path单位: Å, Energy单位: eV
data_series = [
    {
        "label": "Path1-OtoO",
        "path": [0.000000,
0.518889,
1.037088,
1.556334,
2.066119,
2.580106,
3.102435



],
        "energy": [0.000000, 
0.041424,
0.133959,
0.181944,
0.139798,
0.046856,
0.087484
],
        "color": "#000000"  # 绿色
    },
   {
        "label": "Path2-OtoTtoO",
        "path": [0.000000,
0.841093,
1.687890,
2.541173,
3.246527,
3.959589,
4.647087


],
        "energy": [0.000000,
0.074808,
0.278137,
0.480095,
0.366900,
0.246843,
0.239482


],
        "color": "#000000"  # 蓝色
    },
    {
        "label": "Path3-OtoTtoO",
        "path": [0.000000,
0.775731,
1.539898,
2.298585,
3.062947,
3.841461,
4.547757


],
        "energy": [0.000000,
0.075837,
0.274532,
0.432045,
0.514176,
0.544985,
0.593938


],
        "color": "#000000"  # 黄色
    },
{
        "label": "Path4-OtoTtoO",
        "path": [0.000000,
0.756981,
1.523219,
2.299790,
3.087537,
3.698348,
4.313466




],
        "energy": [0.000000,
0.069227,
0.146550,
0.081566,
0.221088,
0.169781,
0.147296




],
        "color": "#000000"  # 黄色
    }
]

# 额外标注文本
#inner_text = "1H2O" # 图内的文字
#title_text = "(a)"  # 图左上角的编号

# ========== 绘图逻辑 ==========

# 遍历绘制每一条路径，分别保存
for idx, series in enumerate(data_series):
    # 为每张图创建一个新的画布
    fig, ax = plt.subplots(figsize=(10, 8))
    # 调整边距
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    x = np.array(series["path"])
    y = np.array(series["energy"])
    label = series["label"]
    color = series["color"]
    
    # 1. 绘制散点 (实心点)
    ax.scatter(x, y, color=color, s=120, zorder=5) 

    # 标注最高点（能垒）
    max_idx = np.argmax(y)
    max_energy = y[max_idx]
    max_path = x[max_idx]
    ax.text(max_path , max_energy + 0.02, f"{max_energy:.2f} eV", 
            ha='center', va='bottom', fontsize=30, fontweight='bold', color=color, zorder=6)
    
    # 2. 绘制平滑曲线
    # 使用 PchipInterpolator 可以避免插值产生的过冲（凹点），保持单调区间内的单调性，更适合NEB路径
    try:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        # model = make_interp_spline(x, y, k=3) # 原来的高阶样条插值容易产生不必要的震荡
        model = PchipInterpolator(x, y) 
        y_smooth = model(x_smooth)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label, zorder=4)
    except Exception as e:
        print(f"插值失败 ({label}): {e}, 使用直接连线代替")
        ax.plot(x, y, color=color, linewidth=3, label=label, zorder=4)

    # ========== 装饰与布局 (对每张图分别设置) ==========
    
    # 坐标轴标签 (粗体)
    ax.set_xlabel("Reaction Path (Å)", fontsize=30, fontweight='bold')
    ax.set_ylabel("Relative Energy (eV)", fontsize=30, fontweight='bold')

    # 坐标轴刻度设置
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=6)

    # 设置Y轴范围和刻度 (保持统一，方便对比)
    ax.set_ylim(-0.05, 0.7)
    ax.set_yticks(np.arange(0, 0.7, 0.1))

    # 设置X轴范围 (可以根据需要统一或自适应，这里暂时保持统一以便对比)
#     ax.set_xlim(-0.1, 5.3)
    xmin, xmax = x.min(), x.max()
    span = xmax - xmin
    if span == 0:
        span = 1.0  # 防止全相同数据时除零/无范围
    ax.set_xlim(xmin - 0.05 * span, xmax + 0.1 * span)
    
    ax.tick_params(top=False, right=False)
    
    # 设置刻度标签为粗体
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 图例
#     legend = ax.legend(loc='lower right', frameon=False, fontsize=30, handletextpad=0.4)
    legend = ax.legend(loc='upper left', frameon=False, fontsize=30, handletextpad=0.4)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # 添加左上角内部文字
    #ax.text(0.04, 0.96, inner_text, transform=ax.transAxes, 
    #        fontsize=25, fontweight='bold', va='top', ha='left')

    # 添加左上角外部标题
    #ax.text(-0.12, 1.0, title_text, transform=ax.transAxes, 
    #        fontsize=35, fontweight='bold', va='bottom', ha='right')

    # 保存文件
    # 文件名不包含非法字符
    safe_label = label.replace("(", "").replace(")", "").replace(" ", "_")
    output_name = f"neb_path_diagram_{safe_label}.png"
    plt.savefig(output_name, dpi=600, bbox_inches='tight')
    
    # 清理画布，防止内存堆积（虽然在循环末尾通常不是大问题，但这里有多个figure）
    plt.close(fig) 
    
    print(f"图片已保存为 {output_name}")
