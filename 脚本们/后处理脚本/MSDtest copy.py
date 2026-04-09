#!/usr/bin/env python3
"""
AIMD结果后处理脚本
用于分析 Li₃Y₀.₆₆Bi₀.₃₃Cl₆ 在700K下的AIMD模拟结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入pymatgen
try:
    from pymatgen.io.vasp import Xdatcar, Outcar
    from pymatgen.core import Structure
    from pymatgen.core.trajectory import Trajectory
    from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
    from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph
    PMG_AVAILABLE = True
    print("✅ pymatgen 已成功导入")
except ImportError as e:
    print(f"❌ 导入pymatgen时出错: {e}")
    print("请确保已安装pymatgen: pip install pymatgen")
    PMG_AVAILABLE = False
    sys.exit(1)

class AIMDAnalyzer:
    """AIMD结果分析器"""
    
    def __init__(self, data_dir=None, temperature=700.0):
        """
        初始化分析器
        
        Parameters:
        -----------
        data_dir : str or Path
            数据目录路径，包含XDATCAR和OUTCAR文件
        temperature : float
            模拟温度 (K)
        """
        if data_dir is None:
            # 默认桌面路径
            desktop = Path.home() / "Desktop"
            self.data_dir = desktop
        else:
            self.data_dir = Path(data_dir)
        
        self.temperature = temperature
        self.xdatcar_path = None
        self.outcar_path = None
        self.xdatcar = None
        self.outcar = None
        self.trajectory = None
        self.structures = []
        self.timesteps = []
        self.total_time = 0.0
        self.time_step = 0.0
        self.nsteps = 0
        
        # 查找文件
        self._find_files()
    
    def _find_files(self):
        """查找XDATCAR和OUTCAR文件"""
        print("🔍 查找AIMD结果文件...")
        
        # 查找XDATCAR文件
        xdatcar_files = list(self.data_dir.glob("XDATCAR*"))
        if xdatcar_files:
            self.xdatcar_path = xdatcar_files[0]
            print(f"  找到 XDATCAR: {self.xdatcar_path}")
        else:
            print("❌ 未找到 XDATCAR 文件")
        
        # 查找OUTCAR文件
        outcar_files = list(self.data_dir.glob("OUTCAR*"))
        if outcar_files:
            self.outcar_path = outcar_files[0]
            print(f"  找到 OUTCAR: {self.outcar_path}")
        else:
            print("❌ 未找到 OUTCAR 文件")
        
        if not (self.xdatcar_path and self.outcar_path):
            print("⚠️  请确保文件存在，或手动指定文件路径")
    
    def load_data(self, xdatcar_path=None, outcar_path=None):
        """
        加载AIMD数据
        
        Parameters:
        -----------
        xdatcar_path : str, optional
            手动指定XDATCAR路径
        outcar_path : str, optional
            手动指定OUTCAR路径
        """
        if xdatcar_path:
            self.xdatcar_path = Path(xdatcar_path)
        if outcar_path:
            self.outcar_path = Path(outcar_path)
        
        if not self.xdatcar_path.exists():
            raise FileNotFoundError(f"XDATCAR文件不存在: {self.xdatcar_path}")
        
        if not self.outcar_path.exists():
            raise FileNotFoundError(f"OUTCAR文件不存在: {self.outcar_path}")
        
        print(f"\n📥 加载AIMD数据...")
        
        try:
            # 加载XDATCAR
            print(f"  加载 XDATCAR: {self.xdatcar_path}")
            self.xdatcar = Xdatcar(str(self.xdatcar_path))
            self.structures = self.xdatcar.structures
            self.nsteps = len(self.structures)
            print(f"  轨迹帧数: {self.nsteps}")
            
            # 加载OUTCAR获取时间信息
            print(f"  加载 OUTCAR: {self.outcar_path}")
            self.outcar = Outcar(str(self.outcar_path))
            
            # 获取时间步长
            self.time_step = self._get_timestep()
            self.total_time = self.nsteps * self.time_step
            self.timesteps = np.arange(0, self.total_time, self.time_step)[:self.nsteps]
            
            print(f"  时间步长: {self.time_step:.3f} ps")
            print(f"  总模拟时间: {self.total_time:.2f} ps")
            print(f"  模拟温度: {self.temperature} K")
            
            # 打印结构信息
            if self.structures:
                initial_structure = self.structures[0]
                print(f"\n📊 初始结构信息:")
                print(f"  化学式: {initial_structure.composition.reduced_formula}")
                print(f"  原子数: {len(initial_structure)}")
                print(f"  晶胞参数: {initial_structure.lattice.parameters}")
                
                # 分析元素组成
                elements = {}
                for site in initial_structure:
                    element = site.specie.symbol
                    elements[element] = elements.get(element, 0) + 1
                
                print(f"  元素分布:")
                for elem, count in elements.items():
                    print(f"    {elem}: {count} 个")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据时出错: {e}")
            raise
    
    def _get_timestep(self):
        """从OUTCAR获取时间步长"""
        try:
            # 尝试从OUTCAR读取POTIM
            with open(self.outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 查找POTIM
            import re
            potim_match = re.search(r'POTIM\s*=\s*([\d\.]+)', content)
            if potim_match:
                potim = float(potim_match.group(1))
                # POTIM通常以fs为单位，转换为ps
                return potim / 1000.0
            
            # 如果找不到POTIM，尝试从NBLOCK和POMASS估算
            nblock_match = re.search(r'NBLOCK\s*=\s*(\d+)', content)
            pomass_match = re.search(r'POMASS\s*=\s*([\d\.\s]+)', content, re.MULTILINE)
            
            if nblock_match and pomass_match:
                nblock = int(nblock_match.group(1))
                # 默认时间步长
                return 0.001 * nblock  # 假设每个离子步1fs
            
        except Exception as e:
            print(f"⚠️  无法从OUTCAR获取时间步长: {e}")
        
        # 默认值：1 fs
        return 0.001
    
    def calculate_msd(self, element='Li', max_time=None, smoothed=False):
        """
        计算均方位移(MSD)
        
        Parameters:
        -----------
        element : str
            要分析的元素
        max_time : float, optional
            最大分析时间 (ps)
        smoothed : bool
            是否对MSD进行平滑处理
        """
        print(f"\n📈 计算{element}的均方位移(MSD)...")
        
        if not self.structures:
            raise ValueError("请先加载AIMD数据")
        
        # 筛选指定元素的原子
        element_indices = []
        initial_structure = self.structures[0]
        
        for i, site in enumerate(initial_structure):
            if site.specie.symbol == element:
                element_indices.append(i)
        
        if not element_indices:
            print(f"⚠️  结构中未找到{element}元素")
            return None
        
        print(f"  {element}原子数: {len(element_indices)}")
        
        # 提取轨迹
        positions = []
        for structure in self.structures:
            element_positions = []
            for idx in element_indices:
                element_positions.append(structure.cart_coords[idx])
            positions.append(element_positions)
        
        positions = np.array(positions)  # 形状: (nsteps, n_atoms, 3)
        
        # 计算MSD
        msd_data = []
        msd_error = []
        time_points = []
        
        # 限制分析时间
        if max_time is None:
            max_time = self.total_time
        max_step = int(min(max_time / self.time_step, self.nsteps))
        
        print(f"  分析时间范围: 0 - {max_time:.2f} ps")
        print(f"  分析步数: {max_step}")
        
        # 计算MSD
        for t in range(1, max_step):
            dt = t * self.time_step
            msd_sum = 0.0
            msd_sq_sum = 0.0
            count = 0
            
            for i in range(self.nsteps - t):
                # 计算位移
                dr = positions[i + t] - positions[i]
                # 考虑周期性边界条件
                for atom_idx in range(len(element_indices)):
                    # 简化的PBC处理
                    dr_atom = dr[atom_idx]
                    # 可以添加更复杂的PBC处理
                    dr2 = np.sum(dr_atom**2)
                    msd_sum += dr2
                    msd_sq_sum += dr2**2
                    count += 1
            
            if count > 0:
                msd_avg = msd_sum / count
                msd_std = np.sqrt((msd_sq_sum / count) - msd_avg**2) / np.sqrt(count)
                msd_data.append(msd_avg)
                msd_error.append(msd_std)
                time_points.append(dt)
        
        msd_data = np.array(msd_data)
        time_points = np.array(time_points)
        
        # 可选：平滑处理
        if smoothed and len(msd_data) > 10:
            from scipy import signal
            window_size = min(11, len(msd_data) // 10)
            if window_size % 2 == 0:
                window_size += 1
            msd_data = signal.savgol_filter(msd_data, window_size, 3)
        
        # 拟合扩散系数
        diffusion_coefficient = self._fit_diffusion_coefficient(time_points, msd_data, element)
        
        return {
            'time': time_points,
            'msd': msd_data,
            'error': msd_error if msd_error else None,
            'diffusion_coefficient': diffusion_coefficient,
            'element': element,
            'n_atoms': len(element_indices)
        }
    
    def _fit_diffusion_coefficient(self, time, msd, element):
        """
        从MSD拟合扩散系数
        
        MSD = 6Dt (三维)
        或 MSD = 4Dt (二维)
        或 MSD = 2Dt (一维)
        
        对于固体电解质，通常使用三维扩散
        """
        if len(time) < 10:
            print(f"⚠️  时间点太少，无法拟合{element}的扩散系数")
            return None
        
        # 使用线性拟合MSD ~ time
        from scipy import stats
        
        # 选择线性区域（排除开始的非线性部分）
        start_idx = max(0, len(time) // 4)  # 从1/4处开始
        end_idx = len(time)
        
        if end_idx - start_idx < 5:
            start_idx = 0
        
        time_fit = time[start_idx:end_idx]
        msd_fit = msd[start_idx:end_idx]
        
        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_fit, msd_fit)
        
        # 三维扩散: D = slope / 6
        D = slope / 6.0  # Å²/ps
        D_cm2_s = D * 1e-16  # 转换为 cm²/s (1 Å²/ps = 1e-16 cm²/s)
        
        print(f"\n📊 {element}扩散系数拟合结果:")
        print(f"  斜率: {slope:.4e} Å²/ps")
        print(f"  截距: {intercept:.4e} Å²")
        print(f"  R²值: {r_value**2:.4f}")
        print(f"  三维扩散系数: {D:.4e} Å²/ps")
        print(f"  三维扩散系数: {D_cm2_s:.4e} cm²/s")
        
        return {
            'D_ang2_ps': D,
            'D_cm2_s': D_cm2_s,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'std_err': std_err
        }
    
    def calculate_conductivity(self, diffusion_coefficients=None, structure=None):
        """
        计算离子电导率
        
        使用Nernst-Einstein方程: σ = (nq²D) / (kBT)
        
        Parameters:
        -----------
        diffusion_coefficients : dict
            各元素的扩散系数
        structure : Structure
            晶体结构
        """
        print(f"\n⚡ 计算离子电导率...")
        
        if structure is None and self.structures:
            structure = self.structures[0]
        
        if structure is None:
            raise ValueError("需要晶体结构信息")
        
        if diffusion_coefficients is None:
            print("⚠️  未提供扩散系数，将自动计算主要迁移离子的扩散系数")
            # 默认分析Li离子
            msd_result = self.calculate_msd(element='Li')
            if msd_result and 'diffusion_coefficient' in msd_result:
                diffusion_coefficients = {'Li': msd_result['diffusion_coefficient']}
            else:
                raise ValueError("无法获取扩散系数")
        
        # 物理常数
        kB = 8.617333262e-5  # eV/K
        q = 1.60217662e-19  # C (电子电荷)
        NA = 6.02214076e23  # 阿伏伽德罗常数
        
        # 计算单位转换因子
        # σ (S/cm) = (n * q² * D) / (kB * T)
        # 其中 n 是载流子浓度 (cm⁻³)
        
        conductivities = {}
        
        for element, D_data in diffusion_coefficients.items():
            if D_data is None:
                continue
            
            D_cm2_s = D_data.get('D_cm2_s')
            if D_cm2_s is None:
                continue
            
            # 计算载流子浓度
            # 对于Li₃Y₀.₆₆Bi₀.₃₃Cl₆，假设所有Li都是可迁移的
            element_count = 0
            for site in structure:
                if site.specie.symbol == element:
                    element_count += 1
            
            # 计算单位体积的载流子数
            volume_cm3 = structure.volume * 1e-24  # Å³ -> cm³
            n_cm3 = element_count / volume_cm3  # cm⁻³
            
            # 计算电导率
            sigma = (n_cm3 * (q**2) * D_cm2_s) / (kB * self.temperature)
            # σ 现在是 S/cm
            
            conductivities[element] = {
                'sigma_S_cm': sigma,
                'sigma_mS_cm': sigma * 1000,  # mS/cm
                'carrier_concentration_cm3': n_cm3,
                'D_cm2_s': D_cm2_s,
                'n_atoms': element_count,
                'volume_cm3': volume_cm3
            }
            
            print(f"\n📊 {element}离子电导率:")
            print(f"  载流子浓度: {n_cm3:.4e} cm⁻³")
            print(f"  扩散系数: {D_cm2_s:.4e} cm²/s")
            print(f"  电导率: {sigma:.4e} S/cm")
            print(f"  电导率: {sigma*1000:.4e} mS/cm")
            print(f"  电导率: {sigma*10000:.4f} μS/cm")
        
        # 计算总电导率（如果多种离子）
        if len(conductivities) > 1:
            total_sigma = sum([data['sigma_S_cm'] for data in conductivities.values()])
            print(f"\n📈 总离子电导率: {total_sigma:.4e} S/cm")
            print(f"                {total_sigma*1000:.4e} mS/cm")
            print(f"                {total_sigma*10000:.4f} μS/cm")
            conductivities['total'] = {
                'sigma_S_cm': total_sigma,
                'sigma_mS_cm': total_sigma * 1000
            }
        
        return conductivities
    
    def plot_msd(self, msd_results, output_dir=None):
        """
        绘制MSD曲线
        
        Parameters:
        -----------
        msd_results : dict or list
            MSD计算结果
        output_dir : str, optional
            输出目录
        """
        if output_dir is None:
            output_dir = self.data_dir / "analysis_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n🎨 绘制MSD曲线...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if not isinstance(msd_results, list):
            msd_results = [msd_results]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(msd_results)))
        
        # MSD vs Time
        ax1 = axes[0]
        for i, result in enumerate(msd_results):
            if result is None:
                continue
            
            time = result['time']
            msd = result['msd']
            element = result.get('element', f'Element {i}')
            diffusion_data = result.get('diffusion_coefficient', {})
            
            # 绘制MSD曲线
            ax1.plot(time, msd, 'o-', markersize=3, linewidth=2, 
                    color=colors[i], label=f'{element} MSD')
            
            # 绘制拟合线
            if diffusion_data and 'slope' in diffusion_data:
                fit_line = diffusion_data['intercept'] + diffusion_data['slope'] * time
                ax1.plot(time, fit_line, '--', linewidth=1.5, 
                        color=colors[i], alpha=0.7, label=f'{element} Fit')
        
        ax1.set_xlabel('Time (ps)', fontsize=12)
        ax1.set_ylabel('MSD (Å²)', fontsize=12)
        ax1.set_title(f'Mean Squared Displacement at {self.temperature}K', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # MSD Log-Log图
        ax2 = axes[1]
        for i, result in enumerate(msd_results):
            if result is None:
                continue
            
            time = result['time']
            msd = result['msd']
            element = result.get('element', f'Element {i}')
            
            ax2.loglog(time, msd, 'o-', markersize=3, linewidth=2, 
                      color=colors[i], label=f'{element}')
            
            # 添加参考线
            if i == 0:
                # 添加斜率为1的参考线
                x_ref = np.array([time[1], time[-1]])
                y_ref = x_ref
                ax2.loglog(x_ref, y_ref, 'k--', alpha=0.5, linewidth=1, label='Slope=1')
        
        ax2.set_xlabel('Time (ps)', fontsize=12)
        ax2.set_ylabel('MSD (Å²)', fontsize=12)
        ax2.set_title('Log-Log Plot of MSD', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"msd_analysis_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  MSD曲线已保存: {output_file}")
        
        plt.show()
        
        return output_file
    
    def plot_conductivity_summary(self, conductivities, output_dir=None):
        """绘制电导率汇总图"""
        if output_dir is None:
            output_dir = self.data_dir / "analysis_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n🎨 绘制电导率汇总图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 柱状图：各离子电导率
        ax1 = axes[0]
        elements = []
        sigma_values = []
        
        for element, data in conductivities.items():
            if element == 'total':
                continue
            if 'sigma_S_cm' in data:
                elements.append(element)
                sigma_values.append(data['sigma_S_cm'])
        
        if elements:
            bars = ax1.bar(elements, sigma_values, color=plt.cm.Set1(np.arange(len(elements))/len(elements)))
            ax1.set_ylabel('Ionic Conductivity (S/cm)', fontsize=12)
            ax1.set_title(f'Ionic Conductivity at {self.temperature}K', fontsize=14)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for bar, val in zip(bars, sigma_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{val:.2e}', ha='center', va='bottom', fontsize=9)
        
        # 总电导率
        ax2 = axes[1]
        if 'total' in conductivities:
            total_sigma = conductivities['total']['sigma_S_cm']
            
            # 创建仪表图
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax2.plot(theta, r, 'k-', linewidth=2)
            
            # 标记常见电导率范围
            common_ranges = {
                'Poor (<10⁻⁶ S/cm)': 1e-6,
                'Moderate (10⁻⁶-10⁻³)': 1e-4,
                'Good (10⁻³-10⁻¹)': 1e-2,
                'Excellent (>0.1 S/cm)': 0.1
            }
            
            for label, value in common_ranges.items():
                angle = value * np.pi / 0.2  # 缩放
                if angle <= np.pi:
                    ax2.plot([angle, angle], [0.9, 1.1], 'k--', alpha=0.3)
                    ax2.text(angle, 1.15, label, ha='center', fontsize=8, rotation=-45)
            
            # 当前电导率位置
            current_angle = min(total_sigma * np.pi / 0.2, np.pi)
            ax2.plot([0, current_angle], [0, 0], 'r-', linewidth=3)
            ax2.plot(current_angle, 0, 'ro', markersize=10)
            
            ax2.set_xlim(0, np.pi)
            ax2.set_ylim(-0.2, 1.3)
            ax2.set_aspect('equal')
            ax2.axis('off')
            ax2.set_title(f'Total Conductivity: {total_sigma:.2e} S/cm\n{total_sigma*1000:.2e} mS/cm', 
                         fontsize=12, pad=20)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"conductivity_summary_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  电导率汇总图已保存: {output_file}")
        
        plt.show()
        
        return output_file
    
    def export_results(self, msd_results, conductivities, output_dir=None):
        """导出分析结果到文件"""
        if output_dir is None:
            output_dir = self.data_dir / "analysis_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 导出MSD数据
        msd_dataframes = []
        
        for i, result in enumerate(msd_results if isinstance(msd_results, list) else [msd_results]):
            if result is None:
                continue
            
            df = pd.DataFrame({
                'Time_ps': result['time'],
                f'MSD_{result.get("element", f"Element_{i}")}_A2': result['msd']
            })
            
            if result.get('error'):
                df[f'Error_{result.get("element", f"Element_{i}")}_A2'] = result['error']
            
            msd_dataframes.append(df)
        
        if msd_dataframes:
            # 合并所有MSD数据
            msd_combined = msd_dataframes[0]
            for df in msd_dataframes[1:]:
                msd_combined = pd.merge(msd_combined, df, on='Time_ps', how='outer')
            
            msd_file = output_dir / f"msd_data_{timestamp}.csv"
            msd_combined.to_csv(msd_file, index=False)
            print(f"📁 MSD数据已导出: {msd_file}")
        
        # 2. 导出扩散系数
        diffusion_data = []
        for i, result in enumerate(msd_results if isinstance(msd_results, list) else [msd_results]):
            if result is None or 'diffusion_coefficient' not in result:
                continue
            
            D_data = result['diffusion_coefficient']
            if D_data:
                diffusion_data.append({
                    'Element': result.get('element', f'Element_{i}'),
                    'D_A2_ps': D_data.get('D_ang2_ps', 'N/A'),
                    'D_cm2_s': D_data.get('D_cm2_s', 'N/A'),
                    'Slope': D_data.get('slope', 'N/A'),
                    'R_squared': D_data.get('r_squared', 'N/A'),
                    'n_atoms': result.get('n_atoms', 'N/A')
                })
        
        if diffusion_data:
            diffusion_df = pd.DataFrame(diffusion_data)
            diffusion_file = output_dir / f"diffusion_coefficients_{timestamp}.csv"
            diffusion_df.to_csv(diffusion_file, index=False)
            print(f"📁 扩散系数已导出: {diffusion_file}")
        
        # 3. 导出电导率
        if conductivities:
            conductivity_data = []
            for element, data in conductivities.items():
                conductivity_data.append({
                    'Element': element,
                    'sigma_S_cm': data.get('sigma_S_cm', 'N/A'),
                    'sigma_mS_cm': data.get('sigma_mS_cm', 'N/A'),
                    'carrier_concentration_cm3': data.get('carrier_concentration_cm3', 'N/A'),
                    'D_cm2_s': data.get('D_cm2_s', 'N/A'),
                    'n_atoms': data.get('n_atoms', 'N/A'),
                    'volume_cm3': data.get('volume_cm3', 'N/A')
                })
            
            conductivity_df = pd.DataFrame(conductivity_data)
            conductivity_file = output_dir / f"conductivity_{timestamp}.csv"
            conductivity_df.to_csv(conductivity_file, index=False)
            print(f"📁 电导率数据已导出: {conductivity_file}")
        
        # 4. 生成汇总报告
        report_file = output_dir / f"analysis_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AIMD 分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"体系: Li₃Y₀.₆₆Bi₀.₃₃Cl₆\n")
            f.write(f"温度: {self.temperature} K\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("模拟参数:\n")
            f.write(f"  总模拟时间: {self.total_time:.2f} ps\n")
            f.write(f"  时间步长: {self.time_step:.3f} ps\n")
            f.write(f"  总步数: {self.nsteps}\n\n")
            
            f.write("扩散系数:\n")
            for data in diffusion_data:
                f.write(f"  {data['Element']}:\n")
                f.write(f"    D = {data.get('D_cm2_s', 'N/A'):.4e} cm²/s\n")
                f.write(f"    R² = {data.get('R_squared', 'N/A'):.4f}\n\n")
            
            f.write("电导率:\n")
            for data in conductivity_data:
                if data['Element'] == 'total':
                    f.write(f"  总电导率: {data.get('sigma_S_cm', 'N/A'):.4e} S/cm\n")
                    f.write(f"            {data.get('sigma_mS_cm', 'N/A'):.4e} mS/cm\n")
                else:
                    f.write(f"  {data['Element']}离子电导率: {data.get('sigma_S_cm', 'N/A'):.4e} S/cm\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"📄 分析报告已生成: {report_file}")
        
        return {
            'msd_file': msd_file if 'msd_file' in locals() else None,
            'diffusion_file': diffusion_file if 'diffusion_file' in locals() else None,
            'conductivity_file': conductivity_file if 'conductivity_file' in locals() else None,
            'report_file': report_file
        }


def main():
    """主函数"""
    print("=" * 60)
    print("AIMD 后处理分析工具")
    #print("体系: Li₃Y₀.₆₆Bi₀.₃₃Cl₆ @ 700K")
    print("=" * 60)
    
    # 设置工作目录为桌面
    desktop = Path.home() / "Desktop"
    print(f"工作目录: {desktop}")
    
    # 创建分析器
    analyzer = AIMDAnalyzer(data_dir=desktop, temperature=1600.0)
    
    try:
        # 加载数据
        if not analyzer.load_data():
            print("\n❌ 数据加载失败")
            return
        
        # 分析迁移离子
        elements_to_analyze = ['Li']  # 主要分析Li离子
        
        msd_results = []
        diffusion_coefficients = {}
        
        for element in elements_to_analyze:
            print(f"\n{'='*60}")
            print(f"分析 {element} 离子扩散")
            print(f"{'='*60}")
            
            # 计算MSD
            msd_result = analyzer.calculate_msd(element=element, max_time=20.0, smoothed=True)
            msd_results.append(msd_result)
            
            if msd_result and 'diffusion_coefficient' in msd_result:
                diffusion_coefficients[element] = msd_result['diffusion_coefficient']
        
        # 绘制MSD曲线
        if msd_results:
            analyzer.plot_msd(msd_results)
        
        # 计算电导率
        if diffusion_coefficients:
            conductivities = analyzer.calculate_conductivity(
                diffusion_coefficients=diffusion_coefficients,
                structure=analyzer.structures[0] if analyzer.structures else None
            )
            
            # 绘制电导率汇总
            if conductivities:
                analyzer.plot_conductivity_summary(conductivities)
            
            # 导出所有结果
            export_files = analyzer.export_results(msd_results, conductivities)
            
            print(f"\n{'='*60}")
            print("🎉 分析完成!")
            print(f"{'='*60}")
            print(f"所有结果已保存到: {desktop / 'analysis_results'}")
            print(f"\n主要结果文件:")
            for key, filepath in export_files.items():
                if filepath:
                    print(f"  {key}: {filepath}")
        
    except Exception as e:
        print(f"\n❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n按Enter键退出...")
    input()


if __name__ == "__main__":
    # 检查pymatgen是否可用
    if not PMG_AVAILABLE:
        print("请先安装pymatgen: pip install pymatgen")
        sys.exit(1)
    
    main()