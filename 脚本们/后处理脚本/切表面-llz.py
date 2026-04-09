import numpy as np  # 导入numpy库用于数值计算
from pathlib import Path  # 导入Path库方便进行路径操作
from ase.geometry import get_layers  # 从ase.geometry导入get_layers，用于分析原子层信息
from ase.constraints import FixAtoms  # 从ase.constraints导入FixAtoms，用于固定原子坐标
import ase.build  # 导入ase.build模块，包含构建晶体和表面的工具
import ase.io  # 导入ase.io模块，用于读写结构文件


def cut_slab(bulk_structure, vector_a, vector_b, nlayers, vacuum=15, **kwargs):
    """
    根据给定的基矢量和层数切割晶体表面。
    
    参数:
    bulk_structure: 体相结构 (Atoms对象)
    vector_a: 定义表面的第一个基矢量 (在原晶包基矢坐标系下)
    vector_b: 定义表面的第二个基矢量
    nlayers: 切割的层数
    vacuum: 真空层总厚度 (Å), 默认为15Å
    **kwargs: 传递给 ase.build.cut 的其他参数 (如 origo)
    """
    # 使用 ase.build.cut 进行切面操作
    # ase.build.cut 允许用户显式指定两个表面矢量 (a, b) 和层数
    # 这比 ase.build.surface 更灵活，因为它不强制要求最小原胞
    slab_structure = ase.build.cut(
        bulk_structure,
        a=vector_a,
        b=vector_b,
        nlayers=nlayers,
        **kwargs
    )
    # 对切出来的结构进行排序，通常按化学式或Z坐标排序，方便后续处理
    sorted_slab_structure = ase.build.sort(slab_structure)
    # 添加真空层。注意：center(vacuum=v) 会在上下各加 v，所以这里除以 2 以保证总真空为 `vacuum`
    # axis=2 表示沿 Z 轴方向添加真空
    sorted_slab_structure.center(vacuum=vacuum / 2, axis=2)
    return sorted_slab_structure


def fix_layers(slab_structure, nlayers=3):
    """
    固定 Slab 底部的若干层原子。
    
    参数:
    slab_structure: 表面结构对象
    nlayers: 需要固定的层数 (从底部开始数)
    """
    _slab_st = slab_structure.copy()  # 复制结构，避免修改原对象
    # get_layers 获取每个原子所属的层索引 (tags)
    # (0, 0, 1) 表示沿着 z 轴方向判断层
    tags = get_layers(_slab_st, (0, 0, 1))
    
    # 获取所有层索引小于 nlayers 的原子索引 (即底部的原子)
    # 注意：get_layers 返回的层索引通常是从底部 (0) 开始递增
    all_fix_idxs = [i for i, tag in enumerate(tags) if tag < nlayers]
    
    # 创建 FixAtoms 约束对象
    fix_ = FixAtoms(indices=all_fix_idxs)
    # 将约束应用到结构上
    _slab_st.set_constraint(fix_)
    return _slab_st


# 向量定义字典：记录不同晶面 (键) 对应的两个基矢量 (值)
# 例如 (1, 1, 1) 面对应的两个矢量可能是 (1, -1, 0) 和 (1, 1, -2)
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

# 读写文件路径配置
# r'' 表示原始字符串，防止路径中的反斜杠转义
bulk_structure = ase.io.read(r'D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\Bi\CONTCAR_Bi3_CONTCAR') # 读取体相文件
outdir = Path(r'D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\Bi') # 输出目录
outdir.mkdir(parents=True, exist_ok=True) # 如果目录不存在则创建

# ===== 第一个切割任务 =====
# 定义本次切割的目标晶面方向 (仅作为文件名标记使用，实际切割由 vector_a/b 控制)
vector = (1, 0, 0)
# 定义切割平面的两个基矢量 a 和 b
# 这里的坐标是相对于原来体相晶胞基矢量的系数
vector_a, vector_b =  (0, 1, 0), (0, 0, 1) # 自定义的一组矢量
nlayers = 14 # 定义切片的层数 (注意：cut 函数的 nlayers 定义可能与 surface 不同，指沿第三矢量堆叠的单胞数)
selected_formula = None

# 循环遍历 origo (原点位移) 参数
# 很多复杂的晶体结构，原子在 z 方向的分布不是均匀的，
# 改变切面的起始原点 (origo) 会切出不同原子终结面 (Termination) 的 Slab
for or_ in np.arange(-0.5, 0.5, 0.01):
    # 调用自定义的切割函数
    slab_structure = cut_slab(
        bulk_structure,
        vector_a,
        vector_b,
        nlayers,
        origo=(or_, 0, or_/ 3) # 设置切割原点，这里让 z 方向和 x 方向联动变化
    )
    # 获取当前切出的 Slab 的化学式
    # mode='hill' 输出紧凑的化学式 (如 Li21Y7Cl42)，方便阅读和筛选
    formula = slab_structure.get_chemical_formula(mode='hill')
    selected_formula = formula
    print(f"origo={or_:.3f}, Formula={formula}") # 打印调试信息

    # 筛选条件：找到特定化学配比的 Slab 时停止寻找
    # 这通常是为了保证切出的表面是化学计量比平衡的，或者是特定的非极性面
    # 注意：ASE Hill表示法的元素顺序可能不同，这里列出可能的组合
    if formula == 'BiCl18Li9Y2' or formula == 'Bi2Cl36Li18Y4' or formula == 'Bi3Cl54Li27Y6':
        print(f"Target found: {formula}")
        break

# 将筛选好的结构写入 VASP 文件
# 输出文件名加入脚本名和 nlayers，便于批量任务区分来源和参数
script_name = Path(__file__).stem
formula_tag = selected_formula if selected_formula else "noformula"
output_name = f"{'_'.join(map(str, vector))}_nlayers{nlayers}_{formula_tag}.vasp"
ase.io.write(outdir / output_name,
             slab_structure, sort=True, direct=True)