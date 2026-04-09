"""
NEB路径图绘制脚本
仿照提供的图片格式绘制 Reaction Path vs Relative Energy
需要安装 scipy: pip install scipy
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

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
        "label": "LIC-(010)",
        "path": [0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0],
        "energy": [0, 0.25, 1.05, 1.02, 1.1, 0.85, 0.75],
        "color": "#000000"  # 绿色
    },
    '''
    {
        "label": "LIC-(100)",
        "path": [0, 1.5, 3.0, 4.5, 6.5, 8.0, 9.5, 11.0],
        "energy": [0, 0.45, 1.6, 1.7, 1.58, 1.45, 1.35, 1.1],
        "color": "#4472C4"  # 蓝色
    },
    {
        "label": "LIC-(111)",
        "path": [0, 2.0, 4.5, 6.5, 7.5, 9.5, 12, 15, 18, 20.5],
        "energy": [0, 0.1, 0.58, 1.45, 1.5, 1.38, 1.3, 1.35, 1.2, 0.98],
        "color": "#FFC000"  # 黄色
    }'''
]

# 额外标注文本
#inner_text = "1H2O" # 图内的文字
#title_text = "(a)"  # 图左上角的编号

# ========== 绘图逻辑 ==========
fig, ax = plt.subplots(figsize=(10, 8))
# 调整边距以适应大字体
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

# 遍历绘制每一条路径
for series in data_series:
    x = np.array(series["path"])
    y = np.array(series["energy"])
    label = series["label"]
    color = series["color"]
    
    # 1. 绘制散点 (实心点)
    ax.scatter(x, y, color=color, s=120, zorder=5) # s是点的大小
    
    # 2. 绘制平滑曲线 (使用样条插值)
    # 避免x点过少导致插值报错，至少需要4个点来进行k=3的样条插值
    k_value = 3 if len(x) > 3 else (2 if len(x) > 2 else 1)
        
    try:
        # 生成平滑的X轴数据
        x_smooth = np.linspace(x.min(), x.max(), 300)
        # 使用 make_interp_spline 创建平滑曲线
        model = make_interp_spline(x, y, k=k_value)
        y_smooth = model(x_smooth)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label, zorder=4)
    except Exception as e:
        print(f"插值失败 ({label}): {e}, 使用直接连线代替")
        ax.plot(x, y, color=color, linewidth=3, label=label, zorder=4)

# ========== 装饰与布局 ==========

# 坐标轴标签 (粗体)
ax.set_xlabel("Reaction Path (Å)", fontsize=30, fontweight='bold')
ax.set_ylabel("Relative Energy (eV)", fontsize=30, fontweight='bold')

# 坐标轴刻度设置
ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=6)

# 设置Y轴范围和刻度 (根据参考图)
ax.set_ylim(-0.1, 2.0)
ax.set_yticks(np.arange(0, 2.1, 0.5)) # 0, 0.5, 1.0, 1.5, 2.0

# 设置X轴范围
ax.set_xlim(-0.1, 21)
# 如果想要像第一张图样隐藏顶部和右侧刻度（但保留边框），matplotlib默认就是这样，或者可以显式设置：
ax.tick_params(top=False, right=False)

# 图例 (Legend) - 右下角，无边框
# handletextpad调整线条和文字的距离
legend = ax.legend(loc='lower right', frameon=False, fontsize=20, handletextpad=0.4)
# 设置图例文本为粗体
for text in legend.get_texts():
    text.set_fontweight('bold')

# 添加左上角内部文字 (e.g. 1H2O)
ax.text(0.04, 0.96, inner_text, transform=ax.transAxes, 
        fontsize=25, fontweight='bold', va='top', ha='left')

# 添加左上角外部标题 (e.g. (a))
# 位置根据图稍作调整
ax.text(-0.12, 1.0, title_text, transform=ax.transAxes, 
        fontsize=35, fontweight='bold', va='bottom', ha='right')

# 保存
output_name = "neb_path_diagram.png"
plt.savefig(output_name, dpi=600, bbox_inches='tight')
plt.show()

print(f"图片已保存为 {output_name}")
