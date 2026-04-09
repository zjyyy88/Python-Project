import numpy as np
from ase.io import read, write
from ase.build import surface
from ase.geometry import cell_to_cellpar

def create_custom_surface(
    input_file='CONTCAR', 
    output_file='POSCAR_surface.vasp', 
    indices=(1, 0, 0), 
    layers=4, 
    vacuum=10.0, 
    vacuum_axis=2, 
    periodic_z=False,
    z_align='center'
):
    """
    按照ASE原本逻辑切表面，并提供自定义真空层和查看化学式的功能。
    
    参数:
    input_file: 输入结构文件名
    output_file: 输出结构文件名
    indices: 晶面指数 (h, k, l)
    layers: 原子层数
    vacuum: 真空层厚度 (Angstrom)。注意：ASE的center方法会在两侧各添加此厚度的真空（总真空 = 2 * vacuum）。
    vacuum_axis: 添加真空层的轴向 (0=x, 1=y, 2=z)。ASE默认surface通常使得垂直方向为z(2)。
    periodic_z: 是否在垂直方向保持周期性
    z_align: Slab在真空层中的对齐方式。'center' (居中), 'bottom' (对齐0, 真空在顶), 'top' (对齐max, 真空在底)。
    """
    
    # 1. 读取体相结构
    print(f"--- 读取文件: {input_file} ---")
    bulk_atoms = read(input_file)
    print(f"原始化学式: {bulk_atoms.get_chemical_formula()}")

    # 2. 切表面 (核心逻辑)
    # ase.build.surface 会自动寻找合适的 v1, v2 (在表面上的矢量) 和 v3 (垂直表面的矢量)
    # 并进行旋转使 v3 沿 z 轴。
    print(f"--- 切取表面: {indices}, 层数: {layers} ---")
    slab = surface(bulk_atoms, indices, layers)
    
    # 获取切完后的初始基矢量信息
    original_cell = slab.get_cell()
    print("切面后的初始晶胞 (v1, v2, v3):")
    print(original_cell)

    # 3. 设置真空层 (自定义位置)
    # vacuum_axis: 0=x, 1=y, 2=z
    print(f"--- 添加真空层: {vacuum} Å (每侧), 方向轴: {vacuum_axis}, 对齐方式: {z_align} ---")
    
    # 首先使用 center 方法确立晶胞大小（包含真空层的总大小）
    # 注意：center(vacuum=v) 会在两侧各加 v，所以总长度增加了 2v
    slab.center(vacuum=vacuum, axis=vacuum_axis)
    
    # 调整位置
    if z_align == 'center':
        pass # 默认已经是居中
    elif z_align == 'bottom':
        # 将原子移动到坐标轴起始位置 (比如 z=0)
        positions = slab.get_positions()
        min_pos = np.min(positions[:, vacuum_axis])
        shift = -min_pos
        positions[:, vacuum_axis] += shift
        slab.set_positions(positions)
        print(f"已将 Slab 底部对齐至轴 {vacuum_axis} = 0")
    elif z_align == 'top':
        # 将原子移动到坐标轴顶端
        positions = slab.get_positions()
        max_pos = np.max(positions[:, vacuum_axis])
        cell_len = slab.get_cell()[vacuum_axis][vacuum_axis]
        shift = cell_len - max_pos
        positions[:, vacuum_axis] += shift
        slab.set_positions(positions)
        print(f"已将 Slab 顶部对齐至轴 {vacuum_axis} = {cell_len:.4f}")
    
    # 4. 设置周期性边界条件
    
    # 4. 设置周期性边界条件
    # 对于表面模型，通常真空方向是不周期的(False)，另外两个方向是周期的(True)。
    pbc = [True, True, True]
    pbc[vacuum_axis] = periodic_z # 按照用户要求设置真空方向的PBC
    slab.set_pbc(pbc)
    
    # 5. 输出信息
    final_formula = slab.get_chemical_formula(mode='all') # mode='all' 包含所有原子，例如 Cu4O4 而不是 CuO
    print(f"--- 最终结构信息 ---")
    print(f"化学式 (包含所有原子): {final_formula}")
    print(f"原子总数: {len(slab)}")
    print(f"晶胞参数 (a, b, c, alpha, beta, gamma):")
    print(cell_to_cellpar(slab.cell))
    
    # 6. 写入文件
    write(output_file, slab, direct=True) # 使用分数坐标写入
    print(f"结构已写入: {output_file}")


if __name__ == "__main__":
    # 在这里可以修改参数
    # 比如想看 v1, v2 的变化，可以通过观察切面前后的 cell 变化体现
    # ASE的 surface 函数不直接提供修改 v1 v2 选取算法的参数，它是通过最小化表面单元面积自动计算的
    
    create_custom_surface(
        input_file='CONTCAR',
        output_file='POSCAR_custom_surface',
        indices=(1, 0, 0),    # 切 (1, 0, 0) 面
        layers=4,             # 4 层原子
        vacuum=7.5,           # 7.5 Å (单侧), 总真空 15 Å (因为 center 加两侧)
        vacuum_axis=2,        # 沿 Z 轴加真空 (标准做法)
        periodic_z=False,     # Z 方向不周期
        z_align='bottom'      # 'bottom' = 对齐 z=0, 真空在顶部
    )
