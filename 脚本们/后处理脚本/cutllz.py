import tempfile  # 创建临时文件和目录
from pathlib import Path  # 处理文件路径

import ase.build  # ASE的晶体构建工具
import ase.io  # ASE的输入输出功能
import numpy as np  # 数值计算库
from ase.constraints import FixAtoms  # 固定原子的约束
from ase.visualize import view  # 原子结构可视化
from atomse.layers import Layers  # 原子分层工具（自定义模块）
from atomse.vesta import vesta  # VESTA可视化接口（自定义模块）

def cut_slab(
    bulk_structure,  # 输入的块体结构
    vector_a,  # 表面a向量
    vector_b,  # 表面b向量
    nlayers,  # 表面层数
    vacuum=20,  # 真空层厚度(Å)
    **kwargs  # 其他可选参数
):
    # 使用ASE的cut函数切割表面
    slab_structure = ase.build.cut(
        bulk_structure,
        a=vector_a,  # 表面a方向
        b=vector_b,  # 表面b方向
        nlayers=nlayers,  # 表面层数
        **kwargs  # 传递额外参数
    )
    # 对原子排序
    sorted_slab_structure = ase.build.sort(slab_structure)
    # 在z方向添加真空层
    sorted_slab_structure.center(vacuum=vacuum / 2, axis=2)
    # 使用临时文件确保格式正确
    with tempfile.NamedTemporaryFile(
        delete=True, suffix='.cif'
    ) as tmp:
        ase.io.write(tmp.name, sorted_slab_structure)  # 写入临时文件
        return ase.io.read(tmp.name)  # 重新读取确保格式一致

# 固定底部原子层
def fix_layers(slab_structure, nlayers=3):  # nlayers: 要固定的层数
    _slab_st = slab_structure.copy()  # 创建副本避免修改原结构
    all_fix_idxs = []  # 存储要固定的原子索引
    # 获取分层信息
    layer_obj = Layers(_slab_st)
    # 遍历前nlayers层
    for idx_ls in layer_obj.layers[:nlayers]:
        all_fix_idxs += idx_ls  # 收集每层原子索引
    # 创建固定约束
    fix_ = FixAtoms(all_fix_idxs)
    # 应用约束
    _slab_st.set_constraint(fix_)
    return _slab_st

# 不同晶面对应的切割向量字典
vectors = {
    (3, 0, 4): ((0, 1, 0), (4 / 3, 2 / 3, -1)),
    (1, 0, 4): ((0, 1, 0), (4 / 3, 2 / 3, -1 / 3)),
    (2, 0, 4): ((4 / 3, 2 / 3, -2 / 3), (0, 1, 0)),
    (1, 1, 1): ((1, -1, 0), (1, 1, -2)),
    (1, 1, 0): ((1, -1, 0), (0, 0, 1)),
    (0, 1, 0): ((1, 0, 0), (0, 0, 1)),
    (1, 0, 0): ((0, 1, 0), (0, 0, 1)),
    (0, 0, 1): ((1, 0, 0), (0, 1, 0)),
    (0, 0, 3): ((1, 0, 0), (0, 1, 0)),
    (0, 2, 1): ((1, 0, 0), (0, 1, -2)),
    (3, 0, 4): ((4 / 3, 0, 2 / 3), (2 / 3, 2 / 3, -1 / 3)),
    (3, 0, 4): ((1 / 3 * 3, 1 / 6 * 3, 0), (0, 3 / 3, -3 / 3)),
    (3, 0, 4): ((0, 3 / 3, -3 / 3), (1 / 3 * 3, 1 / 6 * 3, 0)),
}

# 读取块体结构
bulk_structure_poscar = Path('/home/acduser01/jyZhang/HAW/LIC.vasp')  # POSCAR文件路径
bulk_structure = ase.io.read(bulk_structure_poscar)  # 读取块体结构
outdir = Path('/home/acduser01/slabs/LIC')  # 输出目录

# 定义切割向量 (尝试不同组合)
vector_a, vector_b = (4/3, 4/3, -1/3), (4/3, 0, 1/3)
vector_a, vector_b = (1, 0, 0), (0, 1, 0)
vector_a, vector_b = (1, 0, -1/3), (0, 1, 0)

# 尝试不同原点位置切割表面
vector = (0, 1, 0)  # 目标晶面
nlayers = 7  # 表面层数
# 在[-0.5, 0.4]范围内扫描原点位置
for or_ in np.arange(-0.5, 0.4, 0.01):
    or_ = -0.50020   # 手动设置特定原点值
    # 切割表面结构
    slab_structure = cut_slab(
        bulk_structure,
        vector_a,
        vector_b,
        nlayers,
        origo=(0, 0, or_),  # 设置z方向原点
    )
    # 打印化学式和原点位置
    print(or_, slab_structure.get_chemical_formula())
    #break  # 只执行一次

# 生成文件名
name = '_'.join(map(str, vector))
# 写入VASP格式的结构文件
ase.io.write(
    outdir / f'{name}.vasp',  # 输出路径
    slab_structure,  # 表面结构
    sort=True,  # 原子排序
    direct=True,  # 使用分数坐标
)
"""
# 另一个切割示例 (固定层数不同)
vector = (0, 1, 0)
nlayers = 40
for or_ in np.arange(-0.5, 0.4, 0.01):
    or_ = -0.39  # 特定原点值
    slab_structure = cut_slab(
        bulk_structure,
        vector_a,
        vector_b,
        nlayers,
        extend=1.0,  # 扩展系数
        c=(1, 0, 4),  # c向量
        tolerance=0.1,  # 容差
    )
    print(or_, slab_structure.get_chemical_formula())
    break
# 写入文件
ase.io.write(
    outdir / f'{name}.vasp',
    slab_structure,
    sort=True,
    direct=True,
)

# 另一个晶面切割示例 (0,1,0)
vector = (0, 1, 0)
vector_a, vector_b = np.array(vectors[vector])  # 从字典获取向量
nlayers = 45
for or_ in np.arange(-0.5, 0.4, 0.01):
    or_=0.26  # 特定原点值
    slab_structure = cut_slab(
        bulk_structure,
        vector_a,
        vector_b,
        nlayers,
        tolerance=0.05,  # 更小的容差
        origo=(0, or_, 0)  # y方向原点偏移
    )
    formula = slab_structure.get_chemical_formula()
    print(or_, formula)
    # 检查特定化学式
    if formula == 'Li36In12Cl72':
        break
    break
# 写入到子目录
ase.io.write(
    outdir/'0_1_0' / f'{name}.vasp',
    slab_structure,
    sort=True,
    direct=True,
)
"""
