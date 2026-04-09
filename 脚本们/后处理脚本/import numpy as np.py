import numpy as np
from pymatgen.core import Structure

msd_path = r"E:/固态组/LiLa2O3/势函数/LiLaO2/1600K/msd.dat"  # 改成你的路径
structure_path = r"E:/固态组/LiLa2O3/势函数/LiLaO2/CONTCAR-lammps.vasp"
T = 1600.0  # K
z = 1.0
e = 1.602176634e-19  # C
kB = 1.380649e-23    # J/K

data = np.loadtxt(msd_path)
t_ps = data[:, 0]
msd_a2 = data[:, 1]

# 线性区索引自己调整（例如跳过前 10 点）
i0, i1 = 10, len(t_ps)
k, b = np.polyfit(t_ps[i0:i1], msd_a2[i0:i1], 1)  # Å²/ps

D_a2_ps = k / 6.0
D_cm2_s = D_a2_ps * 1e-4  # 1 Å²/ps = 1e-4 cm²/s

structure = Structure.from_file(structure_path)
n_li = sum(1 for site in structure if site.specie.symbol == "Li")
V_cm3 = structure.volume * 1e-24
n_cm3 = n_li / V_cm3

sigma_S_cm = n_cm3 * (z * e) ** 2 * D_cm2_s / (kB * T)

print(f"slope = {k:.3e} Å²/ps")
print(f"D = {D_cm2_s:.3e} cm²/s")
print(f"σ = {sigma_S_cm:.3e} S/cm  ({sigma_S_cm*1e3:.3e} mS/cm)")