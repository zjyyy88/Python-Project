import os
from bvlain import Lain  # 从bvlain库导入Lain类




file = r'D:\tbb-LPSCI\Li6PS5Cl.cif'  # 正确的结构文件路径
calc = Lain(verbose = True)        # 定义计算器对象，并设置为详细模式
calc.read_file(file) # 读取结构文件

# define params  # 定义参数
params = {
    'mobile_ion': 'Li1+',  # 移动离子类型为Li1+
    'r_cut': 8,          # 截断半径为10.0
    'resolution': 0.5,      # 分辨率为0.2
    'k': 100               # k值为100
}

_ = calc.bvse_distribution(**params) # 计算BVSE分布
energies = calc.percolation_barriers(encut = 5.0)  # 计算渗流势垒，能量截断为3.0
# 安全拼接输出grd文件名，避免覆盖和路径错误
#output_grd = os.path.splitext(file)[0] + '_bvse.grd'
#calc.write_grd(output_grd, task = 'bvse')
calc.write_grd(file)
###计算价态是否与设置的一致
#table = calc.mismatch(r_cut = 4.0)  # 计算失配表，截断半径为4.0
#print(table.to_string())  # 打印表格内容
# 主动释放内存
del calc
gc.collect()