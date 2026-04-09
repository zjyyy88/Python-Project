from ase.io import read, write
from ase.build import surface

atoms = read('CONTCAR')
# Cut surface
s = surface(atoms, (1,0,0), 8)
# Add vacuum to axis 2 (z-direction), which is the default for 'surface' output usually
s.center(vacuum=15, axis=2)

print("New Cell:")
print(s.cell)
write("POSCAR_FIXED", s, direct=True)
print("Write successful")
