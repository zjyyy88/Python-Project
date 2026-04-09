from pymatgen.io.vasp import Vasprun
import numpy as np

v = Vasprun(r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\vasprun.xml')
tdos = v.tdos
max_dos = np.max(tdos.densities[list(tdos.densities.keys())[0]])
min_dos = np.min(tdos.densities[list(tdos.densities.keys())[0]])

print(f"Max DOS value: {max_dos}")
print(f"Min DOS value: {min_dos}")
print(f"Spins: {tdos.densities.keys()}")
