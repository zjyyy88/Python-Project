'''
分析AIMD结果，计算MSD 和 conductivity
'''
import os
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.core import Structure
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
import numpy as np
import pickle
from pymatgen.analysis.diffusion.aimd.pathway import ProbabilityDensityAnalysis
import matplotlib.pyplot as plt


# ---------------------
# 参数设置区
# ---------------------
XDATCAR_PATH = 'E:/固态组/LiLa2O3/LixLayOz-势函数/LiLaO2/pbe/AIMD/XDATCAR-1600K'  # 修改这里为你需要读取的文件路径，可以是绝对路径，例如 r"D:\...\XDATCAR"
TEMPERATURE = 1600
SPECIES = 'Li'
POTIM = 2
NBLOCK = 1
PDA_INTERVAL = 0.5

# 这一步是读取 XDATCAR，得到一系列结构信息
traj = Trajectory.from_file(XDATCAR_PATH)
#traj = Trajectory.from_file('XDATCAR')
# 这一步是实例化 DiffusionAnalyzer 的类
# 并用 from_structures 方法初始化这个类； 900 是温度，2 是POTIM 的值，1是间隔步数
# 间隔步数（step_skip）不太容易理解，但是根据官方教程:
# dt = timesteps * self.time_step * self.step_skip

diff = DiffusionAnalyzer.from_structures(traj, SPECIES, TEMPERATURE, POTIM, NBLOCK)#smoothed=False)
# pda显示离子在晶格中的分布情况，空间网格的步长（单位：Å，埃），表示将整个晶格空间划分成边长为 0.5Å 的立方体网格。
pda = ProbabilityDensityAnalysis.from_diffusion_analyzer(diff,interval=PDA_INTERVAL,species=(SPECIES))
pda.to_chgcar(filename="pda.vasp") 
# 可以用内置的 plot_msd 方法画出 MSD 图像
# 有些终端不能显示图像，这时候可以调用 export_msdt() 方法，得到数据后再自己作图
# 获取 Axes 对象，并通过它的 figure 属性来保存图片
ax = diff.get_msd_plot()
plot_save_path = f"MSD_{SPECIES}_{TEMPERATURE}K.png"
ax.figure.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close(ax.figure)  # 保存后关闭图表，释放内存

diff.export_msdt('msd.dat')
# 接下来直接得到 离子迁移率， 单位是 mS/cm
C = diff.conductivity

with open('result.dat','w') as f:
    f.write('# AIMD result for Li-ion\n')
    f.write('temp\tconductivity\n')
    f.write('%d\t%.2f\n' %(TEMPERATURE, C))


"""

'''IMD结果，计算MSD 和 conductivity
'''
import os
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp.outputs import Xdatcar
from pymatgen.core import Structure
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
import numpy as np
import pickle

# 这一步是读取 XDATCAR，得到一系列结构信息
traj = Trajectory.from_file('XDATCAR')

# 这一步是实例化 DiffusionAnalyzer 的类
# 并用 from_structures 方法初始化这个类； 900 是温度，2 是POTIM 的值，1是间隔步数
# 间隔步数（step_skip）不太容易理解，但是根据官方教程:
# dt = timesteps * self.time_step * self.step_skip

diff = DiffusionAnalyzer.from_structures(traj,'Li',1600,2,5)

# 可以用内置的 plot_msd 方法画出 MSD 图像
# 有些终端不能显示图像，这时候可以调用 export_msdt() 方法，得到数据后再自己作图
#diff.plot_msd()
diff.export_msdt('msd.dat')
'''
###使用msd.dat自己画图
data = np.loadtxt('msd.dat')
    if len(data) > 0 and data.ndim == 2:
        time = data[:, 0]
        msd = data[:, 1]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, msd, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('MSD (Å²)', fontsize=12)
        plt.title('Li Mean Squared Displacement', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('msd_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
'''
# 接下来直接得到 离子迁移率， 单位是 mS/cm
C = diff.conductivity

with open('result.dat','w') as f:
    f.write('# AIMD result for Li-ion\n')
    f.write('temp\tconductivity\n')
    f.write('%d\t%.2f\n' %(1600,C))








"""