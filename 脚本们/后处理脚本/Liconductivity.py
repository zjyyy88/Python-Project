
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

diff = DiffusionAnalyzer.from_structures(traj,'Li',1900,2,1)

# 可以用内置的 plot_msd 方法画出 MSD 图像
# 有些终端不能显示图像，这时候可以调用 export_msdt() 方法，得到数据后再自己作图
#diff.plot_msd()
diff.export_msdt('msd.dat')
###使用msd.dat自己画图

# data = np.loadtxt('msd.dat')
#     if len(data) > 0 and data.ndim == 2:
#         time = data[:, 0]
#         msd = data[:, 1]
        
#         plt.figure(figsize=(10, 6))
#         plt.plot(time, msd, 'b-o', linewidth=2, markersize=4)
#         plt.xlabel('Time (ps)', fontsize=12)
#         plt.ylabel('MSD (Å²)', fontsize=12)
#         plt.title('Li Mean Squared Displacement', fontsize=14)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('msd_plot.png', dpi=300, bbox_inches='tight')
#         plt.show()
# 接下来直接得到 离子迁移率， 单位是 mS/cm

C = diff.conductivity

with open('result.dat','w') as f:
    f.write('# AIMD result for Li-ion\n')
    f.write('temp\tconductivity\n')
    f.write('%d\t%.2f\n' %(1900,C))









