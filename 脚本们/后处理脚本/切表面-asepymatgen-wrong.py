#!/usr/bin/env python
"""
纯结构正交表面切割器
专注于生成正交表面结构，不包含计算参数
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
import warnings
warnings.filterwarnings('ignore')

class OrthogonalSurfaceCutter:
    """
    纯结构正交表面切割器
    只生成结构，不包含计算参数
    """
    
    def __init__(self, structure):
        """
        初始化
        
        参数:
            structure: pymatgen 的 Structure 对象
        """
        self.structure = structure
        self.adaptor = AseAtomsAdaptor()
        
    def get_orthogonal_slabs(self, miller_index, 
                           min_thickness=10.0,
                           min_vacuum=15.0,
                           center_slab=True,
                           max_search=3,
                           orthogonal_tolerance=0.1):
        """
        获取给定米勒指数的所有正交终止面
        
        参数:
            miller_index: 米勒指数，如 (1,1,1)
            min_thickness: 最小平板厚度 (Å)
            min_vacuum: 最小真空层厚度 (Å)
            center_slab: 是否居中平板
            max_search: 最大超胞搜索范围
            orthogonal_tolerance: 正交性容差 (度)
            
        返回:
            list: 正交表面结构列表
        """
        print(f"生成 {miller_index} 表面的正交终止面...")
        
        # 1. 获取所有可能的终止面
        all_slabs = self._get_all_slabs(miller_index, min_thickness, 
                                       min_vacuum, center_slab)
        
        if not all_slabs:
            print(f"未找到 {miller_index} 表面的终止面")
            return []
        
        print(f"找到 {len(all_slabs)} 个不同的终止面")
        
        # 2. 对每个终止面进行正交化
        orthogonal_slabs = []
        
        for i, slab in enumerate(all_slabs):
            print(f"  处理终止面 {i+1}/{len(all_slabs)}...", end="")
            
            # 正交化
            orthogonal_slab = self._orthogonalize_slab(
                slab, max_search, orthogonal_tolerance
            )
            
            if orthogonal_slab is not None:
                orthogonal_slabs.append(orthogonal_slab)
                print(" [OK] 成功正交化")
            else:
                print(" [FAIL] 正交化失败")
        
        print(f"\n[OK] 成功生成 {len(orthogonal_slabs)} 个正交表面")
        return orthogonal_slabs
    
    def _get_all_slabs(self, miller_index, min_thickness, 
                      min_vacuum, center_slab):
        """获取所有可能的终止面"""
        try:
            sg = SlabGenerator(
                self.structure,
                miller_index=miller_index,
                min_slab_size=min_thickness,
                min_vacuum_size=min_vacuum,
                center_slab=center_slab,
                reorient_lattice=False
            )
            return sg.get_slabs(symmetrize=False, repair=False, filter_out_sym_slabs=True)
        except Exception as e:
            print(f"获取终止面时出错: {e}")
            return []
    
    def _orthogonalize_slab(self, slab, max_search, orthogonal_tolerance):
        """
        将slab正交化
        
        核心算法：
        1. 转换为ASE对象
        2. 旋转使表面法线沿z轴
        3. 在xy平面寻找正交基
        4. 重建正交晶胞
        """
        # 1. 转换为ASE对象
        atoms = self.adaptor.get_atoms(slab)
        
        # 2. 获取当前晶格
        cell = atoms.get_cell()
        
        # 检查是否已经正交
        if self._is_orthogonal(cell, orthogonal_tolerance):
            return slab  # 已经是正交，直接返回
        
        # 3. 旋转使表面法线沿z轴
        atoms = self._rotate_surface_to_z(atoms)
        
        # 4. 在xy平面寻找正交基
        atoms, success = self._find_orthogonal_basis(atoms, max_search)
        
        if not success:
            # 如果找不到完美正交基，使用近似正交化
            atoms = self._approximate_orthogonalization(atoms)
        
        # 5. 确保真空层足够
        atoms = self._ensure_vacuum(atoms, min_vacuum=10.0)
        
        # 6. 转换回pymatgen结构
        orthogonal_structure = self.adaptor.get_structure(atoms)
        
        # 添加终止面信息作为属性
        orthogonal_structure.info = {
            'miller_index': slab.miller_index,
            'original_atoms': slab.num_sites,
            'orthogonal_atoms': len(atoms),
            'is_symmetric': getattr(slab, 'is_symmetric', lambda: True)()
        }
        
        return orthogonal_structure
    
    def _is_orthogonal(self, cell, tolerance=0.1):
        """检查晶胞是否正交"""
        angles = self._cell_angles(cell)
        return all(abs(angle - 90) < tolerance for angle in angles)
    
    def _cell_angles(self, cell):
        """计算晶胞角度"""
        a, b, c = cell
        alpha = np.degrees(np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))))
        beta = np.degrees(np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))))
        gamma = np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        return alpha, beta, gamma
    
    def _rotate_surface_to_z(self, atoms):
        """旋转使表面法线沿z轴"""
        cell = atoms.get_cell()
        
        # 假设晶格的第三个向量接近表面法线
        normal = cell[2]
        normal = normal / np.linalg.norm(normal)
        
        z_axis = np.array([0, 0, 1])
        
        # 如果法线不接近z轴，进行旋转
        if abs(np.dot(normal, z_axis)) < 0.999:
            # 计算旋转轴和角度
            rot_axis = np.cross(normal, z_axis)
            rot_axis_norm = np.linalg.norm(rot_axis)
            
            if rot_axis_norm > 1e-6:
                rot_axis = rot_axis / rot_axis_norm
                rot_angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
                
                # 应用旋转
                atoms.rotate(rot_axis, rot_angle, center='COM')
        
        return atoms
    
    def _find_orthogonal_basis(self, atoms, max_search):
        """
        在xy平面寻找正交基
        
        返回:
            (atoms, success): 正交化后的原子和是否成功找到完美正交基
        """
        cell = atoms.get_cell()
        
        # 获取xy平面的晶格向量
        xy_vectors = []
        for i in range(3):
            xy_proj = cell[i, :2]
            if np.linalg.norm(xy_proj) > 1e-3:
                xy_vectors.append((i, cell[i], xy_proj))
        
        if len(xy_vectors) < 2:
            return atoms, False
        
        # 尝试整数组合找到正交基
        best_atoms = None
        best_score = float('inf')
        
        v1_idx, v1, v1_xy = xy_vectors[0]
        v2_idx, v2, v2_xy = xy_vectors[1] if len(xy_vectors) > 1 else xy_vectors[0]
        
        # 搜索整数组合
        for i in range(-max_search, max_search + 1):
            for j in range(-max_search, max_search + 1):
                for k in range(-max_search, max_search + 1):
                    for l in range(-max_search, max_search + 1):
                        if i == 0 and j == 0 and k == 0 and l == 0:
                            continue
                        
                        # 构建新的a,b向量
                        a_new = i * v1 + j * v2
                        b_new = k * v1 + l * v2
                        
                        # 计算正交性
                        cos_angle = abs(np.dot(a_new[:2], b_new[:2]))
                        norm_a = np.linalg.norm(a_new[:2])
                        norm_b = np.linalg.norm(b_new[:2])
                        
                        if norm_a > 1e-6 and norm_b > 1e-6:
                            score = cos_angle / (norm_a * norm_b)
                            
                            if score < best_score:
                                best_score = score
                                
                                # 创建新晶格
                                c_new = np.array([0, 0, cell[2, 2]])
                                new_cell = np.array([
                                    [a_new[0], a_new[1], 0],
                                    [b_new[0], b_new[1], 0],
                                    c_new
                                ])
                                
                                # 复制原子并应用新晶格
                                new_atoms = atoms.copy()
                                new_atoms.set_cell(new_cell, scale_atoms=True)
                                best_atoms = new_atoms
        
        if best_atoms is not None and best_score < 0.01:  # 几乎正交
            return best_atoms, True
        
        return atoms, False
    
    def _approximate_orthogonalization(self, atoms):
        """近似正交化：强制使a沿x轴，b在xy平面与a垂直"""
        cell = atoms.get_cell()
        
        # 使a沿x轴
        a_len = np.linalg.norm(cell[0, :2])
        a_new = np.array([a_len, 0, 0])
        
        # 使b在xy平面，与a垂直
        b_xy = cell[1, :2]
        b_perp = np.array([-b_xy[1], b_xy[0]])
        b_perp = b_perp / np.linalg.norm(b_perp) * np.linalg.norm(b_xy)
        b_new = np.array([b_perp[0], b_perp[1], 0])
        
        # 保持c不变
        c_new = np.array([0, 0, cell[2, 2]])
        
        # 创建新晶格
        new_cell = np.array([a_new, b_new, c_new])
        
        new_atoms = atoms.copy()
        new_atoms.set_cell(new_cell, scale_atoms=True)
        
        return new_atoms
    
    def _ensure_vacuum(self, atoms, min_vacuum=10.0):
        """确保足够的真空层"""
        positions = atoms.get_positions()
        z_coords = positions[:, 2]
        
        z_min, z_max = z_coords.min(), z_coords.max()
        cell_height = atoms.get_cell()[2, 2]
        
        current_vacuum = cell_height - (z_max - z_min)
        
        if current_vacuum < min_vacuum:
            # 增加真空层
            atoms.center(vacuum=min_vacuum, axis=2)
        
        return atoms
    
    def save_structures(self, structures, prefix="slab", format="vasp"):
        """
        保存所有结构
        
        参数:
            structures: 结构列表
            prefix: 文件名前缀
            format: 输出格式 ("vasp", "cif", "xyz", "poscar")
        """
        for i, structure in enumerate(structures):
            filename = f"{prefix}_term{i+1}"
            
            if format.lower() in ["vasp", "poscar"]:
                filename += ".vasp"
                structure.to(filename=filename, fmt="poscar")
            elif format.lower() == "cif":
                filename += ".cif"
                structure.to(filename=filename, fmt="cif")
            elif format.lower() == "xyz":
                filename += ".xyz"
                # 转换为ASE对象保存XYZ
                atoms = self.adaptor.get_atoms(structure)
                from ase.io import write
                write(filename, atoms, format="xyz")
            else:
                print(f"不支持格式: {format}")
                return
            
            print(f"保存终止面 {i+1}: {filename}")


# 使用示例（读取 CONTCAR/POSCAR）
if __name__ == "__main__":
    # 1. 在这里改输入结构路径
    input_file = r"D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\CONTCAR"

    # 2. 在这里改切面和几何参数
    miller_index = (1, 1, 1)
    min_thickness = 10.0
    min_vacuum = 15.0
    center_slab = True

    # 3. 在这里改输出设置
    output_format = "vasp"
    output_prefix = "QJHCONTCAR"

    print("=== 纯结构正交表面切割器 ===\n")
    print(f"读取结构文件: {input_file}")

    structure = Structure.from_file(input_file)
    print(f"化学式: {structure.composition.reduced_formula}")
    print(f"原子数: {structure.num_sites}")

    cutter = OrthogonalSurfaceCutter(structure)
    orthogonal_slabs = cutter.get_orthogonal_slabs(
        miller_index=miller_index,
        min_thickness=min_thickness,
        min_vacuum=min_vacuum,
        center_slab=center_slab,
    )

    if orthogonal_slabs:
        hkl_tag = f"{miller_index[0]}{miller_index[1]}{miller_index[2]}"
        cutter.save_structures(
            orthogonal_slabs,
            prefix=f"{output_prefix}_hkl{hkl_tag}",
            format=output_format,
        )

        print(f"\n{'='*60}")
        print("结果统计:")
        print('='*60)

        for i, slab in enumerate(orthogonal_slabs):
            print(f"\n终止面 {i+1}:")
            print(f"  原子数: {slab.num_sites}")
            print(f"  晶胞参数: {slab.lattice.parameters}")
            print(f"  晶胞角度: {slab.lattice.angles}")
            print(f"  晶胞体积: {slab.lattice.volume:.2f} A^3")

        print("\n[OK] 所有结构已保存")
    else:
        print("[FAIL] 未生成任何正交表面")