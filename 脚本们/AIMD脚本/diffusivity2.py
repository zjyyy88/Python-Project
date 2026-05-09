import numpy as np
import sys
import glob

def calculate_diffusivity(msd_file):
    """
    从 msd.out 文件计算平均扩散系数。

    参数:
    - msd_file (str): 输入文件路径，格式为第一列 frame 数，第二列 MSD (Å²)

    返回:
    - average_diffusivity (float): 平均扩散系数 (Å²/ps)
    """
    # 读取数据文件，略过第一行
    data = np.loadtxt(msd_file, skiprows=1)

    # 提取 frame 数和 MSD 值
    frames = data[:, 0]
    msd_values = data[:, 1]

    # 时间步长 (每帧 0.02 ps)
    time_step = 0.02  # 单位 ps

    # 计算每个帧对应的扩散系数
    diffusivities = msd_values / (6 * time_step * frames)

    # 计算平均扩散系数
    average_diffusivity = np.mean(diffusivities)

    return average_diffusivity


if __name__ == "__main__":
    # 自定义文件名模板
    file_prefix = "msd-"
    custom_list = [300,400,450,500,550,600,800,1000,1200]  # 可自定义文件后缀列表
    file_template = [f"{file_prefix}{num}.out" for num in custom_list]

    # 输出结果
    print("Calculating diffusivity for the following files:")

    for file_name in file_template:
        try:
            avg_diffusivity = calculate_diffusivity(file_name)
            print(f"{file_name}: Average Diffusivity = {avg_diffusivity:.4e} Å²/ps")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
