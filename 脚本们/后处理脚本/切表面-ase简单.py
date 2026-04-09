import ase   
from ase.build import cut,surface # 从ase.build导入切面和构建表面的函数
from ase.visualize import view # 从ase.visualize导入可视化查看器
from ase.io import read,write # 从ase.io导入读取和写入结构文件的函数
Cu = read('CONTCAR') # 读取名为'CONTCAR'的结构文件，并将原子结构赋值给变量Cu
#view(Cu) # 调用可视化查看器显示Cu结构（此行代码被注释，不会执行）
Cusurface = surface(Cu,(1,0,0),2) # 沿(1,0,0)晶面方向切取表面，设置厚度为8层，生成新的表面结构Cusurface
Cusurface.center(vacuum=15,axis=2) # 在z轴方向（axis=2）设置15埃的真空层，并将原子结构居中
write("POSCAR",Cusurface,direct=True) # 将生成的表面结构Cusurface写入名为'POSCAR'的文件中，使用分数坐标（direct=True）