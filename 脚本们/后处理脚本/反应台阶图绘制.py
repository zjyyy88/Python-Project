"""
简单的反应能量台阶图绘制脚本
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
font_size = 25
font_weight = 'normal'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也用类似Times的风格

# ========== 在这里输入你的数据 ==========
# 能量数据（相对于0点的能量值）
#LIC-010
'''
energies = [
            0,
-0.6,
0.157836,
0.447836,
3.077836
]'''
#LIC-100

'''energies = [0,
-0.69,
0.428222,
0.228222,
1.928222,
2.488222]'''
#LIC-111
'''energies = [0,
-0.86,
0.125534,
0.215534,
1.375534,
1.475534]'''

#LYC-100
'''energies = [0,
-1.32,
0.09823419,
0.32823419,
3.00823419,
3.27823419,

]'''
#LYC-110
energies = [
0,
-1.49,
-0.517694,
-0.317694,
1.932306,
1.952306,
]





# 状态标签
#labels = ["surface+H2O", "H2O*", "OH*+H*", "OOH*+H*", "HCl", "O*+H*", "HCl"]
#labels = ["surface+$H_2O^*$", "$H_2O^*$", "$OH^*+H^*$", "HCl","$H^*+O^*$",  "HCl"]#被$包围的部分会被渲染为下标或上标,字体为斜体，_2渲染为₂, ^*渲染为上标*
labels = ["surface+$\\boldsymbol{\mathrm{H_2O^*}}$", "$\\boldsymbol{\mathrm{H_2O^*}}$", "$\\boldsymbol{\mathrm{OH^*+H^*}}$", "HCl","$\\boldsymbol{\mathrm{H^*+O^*}}$",  "HCl"]
# ========== 绑定参数设置 ==========
step_width = 1.0      # 台阶宽度
step_gap = 0.3        # 台阶间距
line_color = 'black'  # 台阶颜色
line_width = 7        # 台阶线宽

# ========== 开始绑定 ==========
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

n = len(energies)
x_positions = [i * (step_width + step_gap) for i in range(n)]

# 绑定台阶和连接线
for i in range(n):
    x = x_positions[i]
    y = energies[i]
    
    # 画水平台阶
    ax.hlines(y=y, xmin=x, xmax=x + step_width, colors=line_color, linewidth=line_width)
    
    # 台阶上方添加标签
    #ax.text(x + step_width/2, y + 0.12, labels[i], ha='center', va='bottom', fontsize=font_size,fontweight=font_weight)
    if i == 0:
        ax.text(x + step_width/2 + 0.2, y + 0.12, labels[i], ha='center', va='bottom', fontsize=font_size, fontweight=font_weight)
    else:
        ax.text(x + step_width/2, y + 0.12, labels[i], ha='center', va='bottom', fontsize=font_size, fontweight=font_weight)
    
    # 台阶下方添加能量值
    #ax.text(x + step_width/2, y - 0.12, f"{y:.2f}", ha='center', va='top', fontsize=18, color='gray')
    
    # 画连接虚线
    if i < n - 1:
        x1 = x + step_width
        x2 = x_positions[i + 1]
        y1 = energies[i]
        y2 = energies[i + 1]
        ax.plot([x1, x2], [y1, y2], 'k--', linewidth=2)
        
        # 标注能量差（放在虚线旁边，不重叠）
        diff = y2 - y1
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        # 根据能量变化方向，将标注放在虚线的上方或下方
        if diff > 0:  # 能量上升，标注放右上方
            offset_x = 0.1
            offset_y = -0.15
            ha= 'left'
        else:  # 能量下降，标注放左下方
            offset_x = - 0.1
            offset_y = -0.05
            ha= 'right'
        ax.text(mid_x + offset_x, mid_y + offset_y, f"{diff:+.2f}", 
                #ha='left',
                ha= ha,
                 va='center', fontsize=font_size, color='red',fontweight=font_weight)

# 画零能量参考线
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_ylim(-1.6, 3.7,)  # Y轴范围从 -1.0 到 3.5
#ax.set_xlim(0, 8.79)
# 设置坐标轴
ax.set_xlabel("Reaction coordinate", fontsize=30,fontweight='bold')
ax.set_ylabel("Relative  energy (eV)", fontsize=30,fontweight='bold')
#ax.set_title("(a) (010)", fontsize=14, loc='left')


# 隐藏x轴刻度，保留边框
ax.set_xticks([])
#ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1]+0.5, 1))
ax.tick_params(axis='y', labelsize=font_size)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.tight_layout()
plt.savefig("energy_diagram.png", dpi=600, bbox_inches='tight')
plt.show()

print("图片已保存为 energy_diagram.png")
