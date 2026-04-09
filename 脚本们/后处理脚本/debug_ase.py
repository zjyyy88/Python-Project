from ase.io import read
import numpy as np

try:
    atoms = read('CONTCAR')
    print("Successfully read CONTCAR")
    print("Cell:")
    print(atoms.cell)
    print("Cell lengths:", atoms.cell.lengths())
    print("Cell angles:", atoms.cell.angles())
    
    from ase.build import surface
    print("Attempting to create surface...")
    s = surface(atoms, (1,0,0), 8)
    print("Surface created.")
    print("Surface Cell:")
    print(s.cell)
except Exception as e:
    print(f"Error: {e}")
