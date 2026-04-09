from bvlain import Lain
import gc

#file =r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\C.cif'
#file = r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\B.cif'
#file = r'E:\固态组\LiLa2O3\LixLa2O3\LiLa2O3-zwbSEND.cif'
#file = r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\Q.cif'
file = r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\c.cif'
calc = Lain(verbose = False)
atoms = calc.read_file(file)       # alternatively, you can use read_atoms() or read_structure()
params = {'mobile_ion': 'Li1+',              # mobile specie
		  'r_cut': 8,                     # cutoff for interaction between the mobile species and framework
		  'resolution': 0.3,	             # distance between the grid points
		  'k': 100,                          # maximum number of neighbors to be collected for each point
          #'use_softbv_covalent_radii': False # default is False, use True to compare results with softBV
}
_ = calc.bvse_distribution(**params)
energies = calc.percolation_barriers(encut = 5.0)
for key in energies.keys():
    print(f'{key[-2:]} percolation barrier is {round(energies[key], 4)} eV')
#calc.write_grd(file + '_bvse', task = 'bvse')  # saves .grd file
calc.write_cube(file + '_bvse', task = 'bvse') # alternatively, save .cube file

# 主动释放内存
del calc
gc.collect()

'''
_ = calc.void_distribution(**params)
radii = calc.percolation_radii()
for key in radii.keys():
    print(f'{key[-2:]} percolation barrier is {round(radii[key], 4)} angstrom')
calc.write_grd(file + '_void', task = 'void') # # save void distribution
'''
