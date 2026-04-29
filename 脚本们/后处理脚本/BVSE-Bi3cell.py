import os
from bvlain import Lain  # 从bvlain库导入Lain类
from pathlib import Path

file = r'D:\tbb-LPSCI\base\1.cif'  # 正确的结构文件路径
print(f"[DEBUG] 输入文件路径: {file}")
print(f"[DEBUG] 文件是否存在: {Path(file).exists()}")

calc = Lain(verbose = True)        # 定义计算器对象，并设置为详细模式
print("[DEBUG] Lain计算器已初始化")
calc.read_file(file) # 读取结构文件
print("[DEBUG] 结构文件已读取")

# define params  # 定义参数
params = {
    'mobile_ion': 'Li1+',  # 移动离子类型为Li1+
    'r_cut': 10.0,          # 截断半径为10.0
    'resolution': 0.2,      # 分辨率为0.2
    'k': 100                # k值为100
}
print(f"[DEBUG] BVSE参数: {params}")

result_bvse = calc.bvse_distribution(**params) # 计算BVSE分布
print(f"[DEBUG] BVSE分布计算完成: {result_bvse}")

energies = calc.percolation_barriers(encut = 5.0)  # 计算渗流势垒，能量截断为3.0
print(f"[DEBUG] 渗流势垒计算完成，共 {len(energies) if energies is not None else 'None'} 条")
print(f"[DEBUG] 渗流势垒数据: {energies}")

# 安全拼接输出grd文件名，避免覆盖和路径错误
output_file = os.path.splitext(file)[0] + '_bvse.grd'
print(f"[DEBUG] 准备输出到: {output_file}")
calc.write_grd(file)
print("[DEBUG] GRD文件已写入")
###计算价态是否与设置的一致
#table = calc.mismatch(r_cut = 4.0)  # 计算失配表，截断半径为4.0
#print(table.to_string())  # 打印表格内容

print("\n" + "="*50)
print("✓ 脚本执行完成!")
print("="*50)