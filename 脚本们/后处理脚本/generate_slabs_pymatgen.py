from pathlib import Path
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

# 1. 设置文件路径 (这里假设读取当前目录下的 CONTCAR，你可以改为 POSCAR 或其他 .cif 路径)
poscar_path = Path('CONTCAR')
if not poscar_path.exists():
    # 如果找不到 CONTCAR，尝试找 POSCAR
    poscar_path = Path('POSCAR')

print(f"Reading structure from: {poscar_path.absolute()}")

# 2. 读取结构
structure = Structure.from_file(poscar_path)

# 3. 设置输出文件夹
outdir = poscar_path.parent / 'slabs'
outdir.mkdir(exist_ok=True)
print(f"Output directory: {outdir.absolute()}")

# 4. 设置切面参数 (仿照截图中的设置)
# 你可以在这里修改为你想要的米勒指数，例如 (1, 0, 0), (1, 1, 1) 等
target_miller_index = (1, 0, 0) 

print(f"Generating slabs for Miller Index: {target_miller_index}")

# 5. 初始化 SlabGenerator
slabgen = SlabGenerator(
    structure,
    miller_index=target_miller_index,
    min_slab_size=30,      # 最小平板厚度 (Angstrom)
    min_vacuum_size=15,    # 最小真空层厚度 (Angstrom)
    center_slab=True,     # 是否将平板居中
    primitive=False,       # 是否寻找原胞
    max_normal_search=10,  #寻找表面法向量的最大搜索范围，若找不到合适的法向量，可以适当增大该值
    lll_reduce=True,      # 是否进行 LLL 约化以获得更正交的晶胞
    in_unit_planes=False, # min_slab_size 是否以原子层数为单位 (False 表示以 Angstrom 为单位)
)

# 6. 获取所有满足条件的 Slabs
all_slabs = slabgen.get_slabs(
    symmetrize=True,      # 是否利用对称性减少计算量
    ftol=0.1,             # 判定对称性的容差
    # repair=True,        # 截图代码中被注释掉的选项
    # max_broken_bonds=3  # 截图代码中被注释掉的选项
)

print(f"Found {len(all_slabs)} slabs.")

# 7. 遍历并保存切片文件
for idx, slab in enumerate(all_slabs, start=1):
    print(f"Saving Slab {idx}...")
    print(f"  Composition: {slab.composition}")
    
    # 构造文件名，例如: 104-001.vasp (对应 (1,0,4) 面，第 1 个构型)
    h, k, l = target_miller_index
    filename = f'{h}{k}{l}-{idx:03d}.vasp'
    output_path = outdir / filename
    
    # 保存为 POSCAR 格式 (通常 .vasp 后缀也是 POSCAR 格式内容)
    slab.to(filename=str(output_path), fmt='POSCAR')
    print(f"  Saved to: {output_path.name}")

print("Done.")
