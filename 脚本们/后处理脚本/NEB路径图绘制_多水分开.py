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
        "label": "LIC-2H2O",
        "path": [0,
0.837628,
1.67388,
2.503453,
3.327616,
4.146009,
4.806225,
5.469756,
6.123202

],
        "energy": [0,
0.069741,
0.352859,
0.905088,
1.163719,
1.285858,
1.271955,
1.276238,
1.235482

],
        "color": "#000000"  # 绿色
    },
    {
        "label": "LIC-3H2O",
        "path": [0,
0.953158,
1.904985,
2.848163,
3.786219,
4.66364,
5.539133

],
        "energy": [0,
0.018918,
0.746847,
1.038011,
1.126961,
0.991415,
0.843881

],
        "color": "#000000"  # 蓝色
    },
    {
        "label": "LIC-4H2O",
        "path": [0,
1.097718,
2.200691,
3.306879,
4.418475,
5.295409,
6.177538

],
        "energy": [0,
0.029871,
0.529078,
1.743566,
2.107543,
2.014269,
1.974106

],
        "color": "#000000"  # 黄色
    },
{
        "label": "LYC-2H2O",
        "path": [0,
0.906203,
1.809162,
2.696868,
3.578515,
4.395751,
5.215233


],
        "energy": [0,
0.023682,
0.289353,
1.547163,
1.892497,
1.726643,
1.618308


],
        "color": "#000000"  # 黄色
    },
{
        "label": "LYC-3H2O",
        "path": [0,
1.010285,
2.010681,
2.991527,
3.958011,
4.964806,
5.967494


],
        "energy": [0,
0.000929,
0.411009,
1.69886,
1.910015,
1.747276,
1.612815


],
        "color": "#000000"  # 黄色
    },
{
        "label": "LYC-4H2O",
        "path": [0,
1.103606,
2.203277,
3.300825,
4.397094,
5.320335,
6.246249



],
        "energy": [0,
0.063689,
0.758916,
1.593435,
1.936447,
1.813492,
1.74683

],
        "color": "#000000"  # 黄色
    }]

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
    ax.text(max_path , max_energy + 0.05, f"{max_energy:.2f} eV", 
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
    ax.set_ylim(-0.2, 2.6)
    ax.set_yticks(np.arange(0, 2.6, 0.5))

    # 设置X轴范围 (可以根据需要统一或自适应，这里暂时保持统一以便对比)
    #ax.set_xlim(-0.1, 21)
    
    ax.tick_params(top=False, right=False)
    
    # 设置刻度标签为粗体
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 图例
    legend = ax.legend(loc='lower right', frameon=False, fontsize=30, handletextpad=0.4)
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
    # dpi=600 已经算是高分辨率了，如果还需要更高，可以改为 1200
    # bbox_inches='tight' 用于去除不必要的白边
    plt.savefig(output_name, dpi=1200)
    
    # 清理画布，防止内存堆积（虽然在循环末尾通常不是大问题，但这里有多个figure）
    plt.close(fig) 
    
    print(f"图片已保存为 {output_name}")
