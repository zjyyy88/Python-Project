#!/usr/bin/env python3
# 文件名：dband_adsorption_english.py
# 功能：生成d-band center与adsorption energy关系的英文静态图

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from adjustText import adjust_text

# 设置英文格式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

# 数据
data = {
    'Element': [
'Ba',
'Ca',
'Sr',
'Ga',
'Nd',
'Pb',
'Sb',
'Sm',
'Sn',
'Al',
'Bi',
'In'
],
    # 'Eads': [-0.780556, -0.745662, 0.018096, 0.395569, 0.080342, 0.08064, -0.48978, -0.10483, -0.537395, 0.325269, -0.432102, -0.375463, 0.027846, -0.457165, -0.584732, -0.446888, 0.00798, 0.140653, -0.344124, -0.416748, -0.186674, -0.498183, 0.08, -0.0396, -0.47229],
    'Eads': [
-0.780556,
-0.745662,
-0.79583,
-0.663154,
-1.137888,
-0.68302,
-0.550347,
-1.107748,
-0.877674,
-0.862752,
-0.730059,
-1.0064358
],
    'p-band': [
-10.156,
-18.709,
-13.967,
2.804,
-17.373,
3.584,
1.216,
-18.67,
2.259,
2.579,
1.572,
3.29
]
}

# 创建DataFrame并清理数据
df = pd.DataFrame(data)
df_clean = df.dropna()

print("Data Processing Information:")
print(f"Total data points: {len(df)}")
print(f"Valid data points: {len(df_clean)}")

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制散点图，按d-band值着色
scatter = ax.scatter(df_clean['p-band'], df_clean['Eads'],
                     c=df_clean['p-band'], cmap='viridis', s=100,
                     alpha=0.8, edgecolors='black', linewidth=0.5)
texts = []
for i, row in df.iterrows():
    text = ax.annotate(row['Element'], (row['p-band'], row['Eads']),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    texts.append(text)

# 自动调整标签位置避免重叠
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
# 添加元素标签
"""for i, row in df_clean.iterrows():
    ax.annotate(row['Element'], 
               (row['p-band'], row['Eads']),
               xytext=(8, 8), 
               textcoords='offset points',
               fontsize=10,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
"""
# 计算趋势线（排除极端值）
x = df_clean['p-band'].values
y = df_clean['Eads'].values

# 过滤掉极端值（d-band < -10）
mask = x > -22
x_filtered = x[mask]
y_filtered = y[mask]

if len(x_filtered) > 1:
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_filtered, y_filtered)

    # 生成趋势线
    x_trend = np.linspace(x_filtered.min(), x_filtered.max(), 100)
    y_trend = slope * x_trend + intercept

    # 绘制趋势线
    # ax.plot(x_trend, y_trend, 'r-', linewidth=3,
    #        label=f'Trend line: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')

    # 在图上添加统计信息
    # stats_text = f'R² = {r_value**2:.4f}\np = {p_value:.4f}'
    # ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
    #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #        verticalalignment='top')

# 添加参考线
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

# 设置坐标轴标签和标题
ax.set_xlabel('p-band Center (eV)', fontsize=14, fontweight='bold')
ax.set_ylabel('Adsorption Energy (eV)', fontsize=14, fontweight='bold')
# ax.set_title('Correlation between d-band Center and Adsorption Energy', fontsize=16, fontweight='bold')

# 添加网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 设置图例
if len(x_filtered) > 1:
    ax.legend(loc='upper right', fontsize=11)

# 调整坐标轴范围，更好地显示数据
ax.set_xlim(-22, 10)
ax.set_ylim(-1.4, -0.2)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('p-band Center (eV)', fontsize=12)

# 添加区域标注
ax.text(0.02, 0.98, 'Weak Adsorption', transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        verticalalignment='top')
ax.text(0.02, 0.02, 'Strong Adsorption', transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
        verticalalignment='bottom')

# 保存图片
plt.tight_layout()
plt.savefig('pband_adsorption_english.png', dpi=300, bbox_inches='tight')
plt.savefig('pband_adsorption_english.pdf', bbox_inches='tight')
plt.savefig('pband_adsorption_english.svg', bbox_inches='tight')

# 显示图形
plt.show()

# 打印统计信息
print("\n=== Statistical Analysis ===")
if len(x_filtered) > 1:
    print(f"Regression equation: y = {slope:.4f}x + {intercept:.4f}")
    print(f"Correlation coefficient R: {r_value:.4f}")
    print(f"Determination coefficient R²: {r_value ** 2:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Standard error: {std_err:.4f}")

print(f"\nData range:")
print(f"p-band: {df_clean['p-band'].min():.2f} to {df_clean['p-band'].max():.2f} eV")
print(f"Adsorption energy: {df_clean['Eads'].min():.3f} to {df_clean['Eads'].max():.3f} eV")

# 保存数据到CSV
df_clean.to_csv('pband_adsorption_data.csv', index=False)
print(f"\nData saved to: pband_adsorption_data.csv")
print("Images saved as: pband_adsorption_english.png, .pdf, .svg")











