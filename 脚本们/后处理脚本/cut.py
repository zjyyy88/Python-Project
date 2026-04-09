import numpy as np
from pathlib import Path
from ase.geometry import get_layers
from ase.constraints import FixAtoms
import ase.build
import ase.io


def cut_slab(bulk_structure, vector_a, vector_b, nlayers, vacuum=15, **kwargs):
    slab_structure = ase.build.cut(
        bulk_structure,
        a=vector_a,
        b=vector_b,
        nlayers=nlayers,
        **kwargs
    )
    sorted_slab_structure = ase.build.sort(slab_structure)
    sorted_slab_structure.center(vacuum=vacuum / 2, axis=2)
    return sorted_slab_structure


def fix_layers(slab_structure, nlayers=3):
    _slab_st = slab_structure.copy()
    tags = get_layers(_slab_st, (0, 0, 1))
    all_fix_idxs = [i for i, tag in enumerate(tags) if tag < nlayers]
    fix_ = FixAtoms(indices=all_fix_idxs)
    _slab_st.set_constraint(fix_)
    return _slab_st


# 向量定义



# 文件路径
bulk_structure = ase.io.read(r'd:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\bulk\LiYCl\QJHCONTCAR-Bi\CONTCAR_Bi3.vasp')
outdir = Path(r'd:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\bulk\LiYCl\QJHCONTCAR-Bi')
outdir.mkdir(parents=True, exist_ok=True)

# ===== 第一个切割任务 =====
#vector = (1, -1, 0)
#vector_a, vector_b =  (0, 1, 0), (4/3, 2/3, -1/3)  # 使用第一个向量组合

vector = (1, 1, 1)
vector_a , vector_b =  (1, -1, 0), (1, 1, -2)
nlayers =6
for or_ in np.arange(-0.5, 0.4, 0.1):
    slab_structure = cut_slab(
        bulk_structure,
        vector_a,
        vector_b,
        nlayers,
        origo=(or_, 0, or_/ 3)
    )
    formula = slab_structure.get_chemical_formula()
    print(f"origo={or_:.3f}, Formula={formula}")

    # 找到目标成分时跳出循环
    if formula == 'Bi1Y2Li9Cl18':
        break

ase.io.write(outdir / f"{'_'.join(map(str, vector))}.vasp",
             slab_structure, sort=True, direct=True)