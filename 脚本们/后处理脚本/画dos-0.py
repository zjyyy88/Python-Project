from pymatgen.io.vasp import Vasprun  # 导入 pymatgen 的 Vasprun 类，用于解析 VASP 输出文件
from pymatgen.electronic_structure.plotter import DosPlotter  # 导入 DosPlotter 类，用于绘制态密度 (DOS) 图
from pymatgen.electronic_structure.core import Spin # 导入 Spin 枚举
import matplotlib.pyplot as plt # 导入 matplotlib 用于自定义绘图
import pandas as pd  # 导入 pandas 库，用于数据处理

# 读取指定路径的 vasprun1.xml 文件，实例化 Vasprun 对象
# 注意：这里读取的是 vasprun1.xml，请确保文件存在
v = Vasprun(r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\vasprun1.xml')

tdos = v.tdos  # 从 Vasprun 对象中获取总态密度 (Total DOS) 数据
plottertdos = DosPlotter()  # 创建一个 DosPlotter 绘图对象
plottertdos.add_dos("Total DOS", tdos)  #添加总态密度数据到绘图对象，图例名称为 "Total DOS"
plottertdos.show(xlim=[-8, 8], ylim=[-150, 150])  # 显示图像，并设置 x 轴范围为 [-8, 8]，y 轴范围为 [-150, 150]
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



plotterelement.show(xlim=[-8, 8], ylim=[-75, 75])  
plotterelement.save_plot(filename='element.png', xlim=[-8, 8], ylim=[-75, 75]) 


cdos = v.complete_dos  # 获取完整的态密度数据 (Complete DOS)
spd_dos = cdos.get_spd_dos()  # 获取分波态密度 (SPD-resolved DOS)，即按 s, p, d 轨道投影的 DOS
plotterguidao = DosPlotter()  # 创建一个新的 DosPlotter 绘图对象
plotterguidao.add_dos_dict(spd_dos)  # 将分波态密度数据字典添加到绘图对象中
plotterguidao.show(xlim=[-8, 8], ylim=[-50, 50])  # 显示图像，并设置 x 轴范围为 [-8, 8]，y 轴范围为 [-50, 50]
plotterguidao.save_plot(filename='guidao.png', xlim=[-8, 8], ylim=[-50, 50])  # 将图像保存为 b.png，自动识别格式，同时设置坐标轴范围
