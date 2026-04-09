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


# 这一步是读取 XDATCAR，得到一系列结构信息
traj = Trajectory.from_file('XDATCAR')

# 这一步是实例化 DiffusionAnalyzer 的类
# 并用 from_structures 方法初始化这个类； 900 是温度，2 是POTIM 的值，1是间隔步数
# 间隔步数（step_skip）不太容易理解，但是根据官方教程:
# dt = timesteps * self.time_step * self.step_skip

diff = DiffusionAnalyzer.from_structures(traj,'Li',800,2,1)#smoothed=False)
# pda显示离子在晶格中的分布情况，空间网格的步长（单位：Å，埃），表示将整个晶格空间划分成边长为 0.5Å 的立方体网格。
pda = ProbabilityDensityAnalysis.from_diffusion_analyzer(diff,interval=0.5,species=("Li"))
pda.to_chgcar(filename="pda.vasp") 
# 可以用内置的 plot_msd 方法画出 MSD 图像
# 有些终端不能显示图像，这时候可以调用 export_msdt() 方法，得到数据后再自己作图
diff.plot_msd()
diff.export_msdt('msd.dat')
# 接下来直接得到 离子迁移率， 单位是 mS/cm
C = diff.conductivity

with open('result.dat','w') as f:
    f.write('# AIMD result for Li-ion\n')
    f.write('temp\tconductivity\n')
    f.write('%d\t%.2f\n' %(800,C))


