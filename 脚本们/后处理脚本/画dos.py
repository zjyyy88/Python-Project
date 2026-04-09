from pymatgen.io.vasp import Vasprun  # 导入 pymatgen 的 Vasprun 类，用于解析 VASP 输出文件
from pymatgen.electronic_structure.plotter import DosPlotter  # 导入 DosPlotter 类，用于绘制态密度 (DOS) 图
from pymatgen.electronic_structure.core import Spin # 导入 Spin 枚举
import matplotlib.pyplot as plt # 导入 matplotlib 用于自定义绘图
import pandas as pd  # 导入 pandas 库，用于数据处理

# 读取指定路径的 vasprun1.xml 文件，实例化 Vasprun 对象
# 注意：这里读取的是 vasprun1.xml，请确保文件存在
v = Vasprun(r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\vasprun.xml')

tdos = v.tdos  # 从 Vasprun 对象中获取总态密度 (Total DOS) 数据
plottertdos = DosPlotter()  # 创建一个 DosPlotter 绘图对象
plottertdos.add_dos("Total DOS", tdos)  #添加总态密度数据到绘图对象，图例名称为 "Total DOS"
# plottertdos.show(xlim=[-8, 8], ylim=[-150, 150])  # 显示图像，并设置 x 轴范围为 [-8, 8]，y 轴范围为 [-150, 150]
#plottertdos.save_plot(filename='a.eps',xlim=[-8, 4], ylim=[-20, 20]) # the default image format is eps
# 将 save_plot 放在 show 之前，或者使用 plt.savefig
# 注意：DosPlotter.save_plot 也会生成一个新的图，所以放在 show 后面前面都可以，但建议使用 png 格式
plottertdos.save_plot(filename='tdos.png', xlim=[-8, 8], ylim=[-150, 150])  # 将图像保存为 tdos.png，自动识别格式，同时设置坐标轴范围


# 获取能带结构对象，避免重复调用 (get_band_structure 比较耗时)
bs = v.get_band_structure()  # 从 Vasprun 对象中获取能带结构 (BandStructure)

# 获取gap/vbm/cbm
bandgap_dict = bs.get_band_gap()  # 获取带隙信息字典 (包括带隙大小、跃迁类型等)
vbm = bs.get_vbm()["energy"]  # 获取价带顶 (VBM) 的能量值
cbm = bs.get_cbm()["energy"]  # 获取导带底 (CBM) 的能量值

# 添加到数据框中
bandgap = pd.DataFrame([bandgap_dict])  # 将带隙字典转换为 DataFrame (单行)
bandgap["VBM"] = vbm  # 添加一列 "VBM"，值为 VBM 能量
bandgap["CBM"] = cbm  # 添加一列 "CBM"，值为 CBM 能量

print(bandgap)  # 打印带隙信息 DataFrame



cdos = v.complete_dos  # 从 Vasprun 对象中获取完整的态密度数据 (Complete DOS)
element_dos = cdos.get_element_dos()  # 获取元素投影的态密度 (Element-resolved DOS)
plotterelement = DosPlotter(sigma=0.01)   # 创建 DosPlotter 对象，sigma 参数用于控制展宽 (smearing)
plotterelement.add_dos_dict(element_dos)  # 将元素态密度数据添加到绘图对象中

# --- 自定义颜色绘图部分开始 ---
# pymatgen 的 DosPlotter 默认不支持按元素指定颜色，所以我们提取数据用 maplotlib 自己画
dos_data = plotterelement.get_dos_dict() # 获取处理过（如 smearing）的数据

# 调试：打印数据信息以检查为何为空
print("正在检查 DOS 数据...")
print("包含的元素:", list(dos_data.keys()))
if not dos_data:
    print("警告：dos_data 为空！请检查 vasprun 文件读取是否正确。")
else:
    first_elem = list(dos_data.keys())[0]
    print(f"元素 {first_elem} 的密度 keys:", dos_data[first_elem]["densities"].keys())
    # 检查数据范围
    max_dens = 0
    for data in dos_data.values():
        for d in data["densities"].values():
            import numpy as np
            max_dens = max(max_dens, np.max(np.abs(d)))
    print(f"最大 DOS 密度值: {max_dens}")

# 在这里设置你想要的颜色
element_colors = {
    "Li": "green",
    "Y": "orange",  # 当前体系包含 Y，给它一个颜色
    "Cl": "blue",
    "In": "orange", #以此类推
    "Bi": "red",
    "O": "red"
}

plt.figure(figsize=(21, 14)) # 创建绘图窗口
#plt.figure()
# 遍历所有元素的数据进行绘制
# label 是元素名称（如 "Li", "Cl"），data 是该元素对应的 DOS 数据字典
for label, data in dos_data.items():
    energies = data["energies"]  # 获取能量轴数据 (x 轴)
    densities = data["densities"]  # 获取态密度值数据 (y 轴)，包含自旋向上和向下
    color = element_colors.get(label, "black") # 获取颜色，如果字典中未定义该颜色的元素，则默认使用黑色
    
    # --- 处理自旋向上 (Spin Up) 的数据 ---
    # 尝试多种 Key 的可能性，因为不同版本的 pymatgen 可能使用不同的键名
    # Spin.up (通常对应 "1" 或 Spin.up 对象)
    y_up = None
    if Spin.up in densities:
        y_up = densities[Spin.up]
    elif str(Spin.up) in densities: # 尝试将枚举转为字符串查找，例如 "Spin.up"
         y_up = densities[str(Spin.up)]
    elif "1" in densities: # 某些旧版本可能使用字符串 "1" 代表自旋向上
        y_up = densities["1"]
    elif 1 in densities: # 或者直接使用整数 1
        y_up = densities[1]

    # 如果找到了自旋向上的数据，则进行绘制
    if y_up is not None:
        # plot(x, y, color, label, linewidth)
        # label 参数用于图例显示，linewidth 控制线条粗细
        plt.plot(energies, y_up, color=color, label=label, linewidth=3)

    # --- 处理自旋向下 (Spin Down) 的数据 ---
    # 自旋向下通常显示在 x 轴下方（负值区域），用于区分自旋方向
    # Spin.down (通常对应 "-1" 或 Spin.down 对象)
    y_down = None
    if Spin.down in densities:
        y_down = densities[Spin.down]
    elif str(Spin.down) in densities:
        y_down = densities[str(Spin.down)]
    elif "-1" in densities:
        y_down = densities["-1"]
    elif -1 in densities:
        y_down = densities[-1]

    # 如果找到了自旋向下的数据
    if y_down is not None:
         # 注意：原始数据通常是正值，为了显示在下方，我们需要取负
         # 如果 y_down 是 list，不能直接用 -y_down，需要列表推导或转 numpy 数组处理
        import numpy as np
        y_down_np = -np.array(y_down) # 将列表转换为 numpy 数组并取负值
        # 绘制自旋向下曲线，颜色与自旋向上保持一致，但不重复添加 label（避免图例重复）
        plt.plot(energies, y_down_np, color=color, linewidth=2)

# --- 设置图形的坐标轴和外观属性 ---
plt.xlim(-8, 8)  # 设置 x 轴显示范围 (eV)
plt.ylim(-75, 75) # 设置 y 轴显示范围，如果数据溢出可调整此数值
plt.xlabel("Energy (eV)", fontsize=18) # 设置 x 轴标签文本及字体大小
plt.ylabel("Density of States(states/eV)", fontsize=18) # 设置 y 轴标签文本及字体大小
plt.tick_params(axis='both', which='major', labelsize=14) # 设置刻度值的字体大小

# 设置边框 (Spines) 粗细
# plt.gca() 获取当前 axes 对象 (Current Axes)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2) # 底部边框加粗
ax.spines['left'].set_linewidth(2)   # 左侧边框加粗
ax.spines['top'].set_linewidth(2)    # 顶部边框加粗
ax.spines['right'].set_linewidth(2)  # 右侧边框加粗

plt.legend(fontsize=14, loc='upper right') # 显示图例，设置字体大小及位置
#plt.title("Element DOS with Custom Colors", fontsize=20) # 设置标题（可选）
plt.axvline(x=0, color="k", linestyle="--", linewidth=1) # 在 x=0 处添加一条黑色虚线，表示费米能级
#plt.grid(True, alpha=0.3) # 添加网格线，透明度为 0.3（可选）

plt.savefig("element_custom_colors.png") # 将当前图形保存为 PNG 文件
plt.show() # 显示图形窗口（如果是在脚本中运行，通常会弹出窗口，在 Notebook 中则直接显示）
print("自定义颜色绘图完成，已保存为 element_custom_colors.png")
# --- 自定义颜色绘图部分结束 ---

# 原来的绘图代码（已被上面的自定义代码替代，可以选择注释掉或删除）
plotterelement.show(xlim=[-8, 8], ylim=[-75, 75])  
plotterelement.save_plot(filename='element.png', xlim=[-8, 8], ylim=[-75, 75]) 


cdos = v.complete_dos  # 获取完整的态密度数据 (Complete DOS)
spd_dos = cdos.get_spd_dos()  # 获取分波态密度 (SPD-resolved DOS)，即按 s, p, d 轨道投影的 DOS
plotterguidao = DosPlotter()  # 创建一个新的 DosPlotter 绘图对象
plotterguidao.add_dos_dict(spd_dos)  # 将分波态密度数据字典添加到绘图对象中
# plotterguidao.show(xlim=[-8, 8], ylim=[-50, 50])  # 显示图像，并设置 x 轴范围为 [-8, 8]，y 轴范围为 [-50, 50]
plotterguidao.save_plot(filename='guidao.png', xlim=[-8, 8], ylim=[-50, 50])  # 将图像保存为 b.png，自动识别格式，同时设置坐标轴范围
