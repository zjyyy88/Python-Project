import numpy as np

# 定义 POSCAR 文件路径
poscar_file = "POSCAR"

# 读取 POSCAR 文件
with open(poscar_file, "r") as f:
    lines = f.readlines()

# 提取晶格基矢量
a = np.array([float(x) for x in lines[2].split()])
b = np.array([float(x) for x in lines[3].split()])
c = np.array([float(x) for x in lines[4].split()])

# 计算晶胞体积 (单位: Å³)
volume_angstrom3 = np.abs(np.dot(a, np.cross(b, c)))

# 转换晶胞体积为 cm³ (1 Å³ = 1e-24 cm³)
volume_cm3 = volume_angstrom3 * 1e-24

# 提取原子种类和数量
atom_types = lines[5].split()
atom_counts = [int(x) for x in lines[6].split()]

# 找到 Li 的数量
if "Li" in atom_types:
    li_index = atom_types.index("Li")  # 获取 Li 的索引
    li_count = atom_counts[li_index]  # 获取 Li 的数量
else:
    li_count = 0  # 如果没有 Li，设为 0

# 计算 Li 浓度 (单位: ions/cm³)
li_concentration = li_count / volume_cm3

# 输出结果
print(f"晶胞体积: {volume_angstrom3:.3f} Å³ ({volume_cm3:.3e} cm³)")
print(f"Li 离子数量: {li_count}")
print(f"Li 离子浓度: {li_concentration:.3e} ions/cm³")
