#!/usr/bin/env python3
"""
pymatgen MD 后处理脚本
用于分析 Li3Y0.66Bi0.33Cl6 体系的分子动力学结果
"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Xdatcar, Outcar
from pymatgen.core import Structure
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.analysis.local_env import RadialDistributionFunction
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class MDPostProcessor:
    """分子动力学后处理器"""
    
    def __init__(self, trajectory_file="XDATCAR", temperature=1200.0):
        self.trajectory_file = trajectory_file
        self.temperature = temperature
        self.trajectory = None
        self.structures = []
        
    def load_trajectory(self):
        """加载MD轨迹"""
        print("📥 加载MD轨迹...")
        try:
            xdatcar = Xdatcar(self.trajectory_file)
            self.structures = xdatcar.structures
            print(f"✅ 成功加载 {len(self.structures)} 帧轨迹")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def calculate_msd(self, element="Li", time_step=1.0):
        """计算均方位移"""
        print(f"\n📈 计算{element}的MSD...")
        
        if not self.structures:
            print("❌ 请先加载轨迹")
            return None
        
        # 获取元素索引
        indices = []
        initial_structure = self.structures[0]
        for i, site in enumerate(initial_structure):
            if site.specie.symbol == element:
                indices.append(i)
        
        if not indices:
            print(f"❌ 未找到{element}元素")
            return None
        
        print(f"🔬 分析{len(indices)}个{element}原子")
        
        # 计算MSD
        n_frames = len(self.structures)
        msd_values = []
        times = []
        
        for dt in range(1, min(100, n_frames)):  # 分析前100个时间间隔
            msd_sum = 0.0
            count = 0
            
            for t0 in range(0, n_frames - dt):
                # 提取两个时间点的坐标
                pos0 = np.array([self.structures[t0].cart_coords[i] for i in indices])
                pos1 = np.array([self.structures[t0+dt].cart_coords[i] for i in indices])
                
                # 计算位移（简化处理，未考虑PBC）
                dr = pos1 - pos0
                dr2 = np.sum(dr**2, axis=1)
                msd_sum += np.mean(dr2)
                count += 1
            
            if count > 0:
                msd_avg = msd_sum / count
                msd_values.append(msd_avg)
                times.append(dt * time_step)
        
        # 拟合扩散系数
        if len(msd_values) > 5:
            coeffs = np.polyfit(times, msd_values, 1)
            D = coeffs[0] / 6.0  # 三维扩散系数
            D_cm2_s = D * 1e-16  # 转换为 cm²/s
            
            print(f"📊 {element}扩散分析:")
            print(f"  D = {D:.4e} Å²/ps")
            print(f"  D = {D_cm2_s:.4e} cm²/s")
            
            return {
                'time': times,
                'msd': msd_values,
                'D_ang2_ps': D,
                'D_cm2_s': D_cm2_s,
                'slope': coeffs[0],
                'intercept': coeffs[1]
            }
        
        return None
    
    def calculate_rdf(self, element1="Li", element2="Cl", frame=0):
        """计算径向分布函数"""
        print(f"\n📊 计算RDF: {element1}-{element2}")
        
        if frame >= len(self.structures):
            frame = 0
        
        structure = self.structures[frame]
        
        # 计算RDF
        rdf_calc = RadialDistributionFunction(structure, r_max=10.0, bin_size=0.1)
        
        # 获取距离列表
        distances = []
        for i, site1 in enumerate(structure):
            if site1.specie.symbol == element1:
                for j, site2 in enumerate(structure):
                    if i != j and site2.specie.symbol == element2:
                        dist = structure.get_distance(i, j)
                        if dist <= 10.0:
                            distances.append(dist)
        
        # 创建直方图
        bins = np.arange(0, 10.0, 0.1)
        hist, bin_edges = np.histogram(distances, bins=bins, density=True)
        
        return {
            'distances': (bin_edges[:-1] + bin_edges[1:]) / 2,
            'rdf': hist,
            'element1': element1,
            'element2': element2
        }
    
    def calculate_conductivity(self, D_cm2_s, structure_file="CONTCAR"):
        """计算离子电导率 (Nernst-Einstein方程)"""
        print(f"\n⚡ 计算离子电导率...")
        
        # 加载结构
        structure = Structure.from_file(structure_file)
        
        # 计算Li离子浓度
        li_count = 0
        for site in structure:
            if site.specie.symbol == "Li":
                li_count += 1
        
        # 计算载流子浓度
        volume_cm3 = structure.volume * 1e-24  # Å³ → cm³
        n = li_count / volume_cm3  # cm⁻³
        
        # 物理常数
        kB = 1.380649e-23  # J/K
        q = 1.60217662e-19  # C
        
        # 计算电导率
        sigma = (n * q**2 * D_cm2_s) / (kB * self.temperature)
        
        print(f"📊 计算结果:")
        print(f"  Li离子数: {li_count}")
        print(f"  体积: {structure.volume:.2f} Å³")
        print(f"  载流子浓度: {n:.4e} cm⁻³")
        print(f"  扩散系数: {D_cm2_s:.4e} cm²/s")
        print(f"  电导率: {sigma:.4e} S/cm")
        print(f"  电导率: {sigma*1000:.4e} mS/cm")
        
        return {
            'sigma_S_cm': sigma,
            'sigma_mS_cm': sigma * 1000,
            'carrier_concentration': n,
            'D_cm2_s': D_cm2_s
        }
    
    def plot_results(self, msd_data, rdf_data, output_file="md_analysis.png"):
        """绘制分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. MSD曲线
        ax1 = axes[0, 0]
        ax1.plot(msd_data['time'], msd_data['msd'], 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Time (ps)', fontsize=12)
        ax1.set_ylabel('MSD (Å²)', fontsize=12)
        ax1.set_title(f'Li MSD at {self.temperature}K', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 添加拟合线
        fit_line = msd_data['intercept'] + msd_data['slope'] * np.array(msd_data['time'])
        ax1.plot(msd_data['time'], fit_line, 'r--', linewidth=2, label=f"D = {msd_data['D_cm2_s']:.2e} cm²/s")
        ax1.legend()
        
        # 2. RDF
        ax2 = axes[0, 1]
        ax2.plot(rdf_data['distances'], rdf_data['rdf'], 'g-', linewidth=2)
        ax2.set_xlabel('Distance (Å)', fontsize=12)
        ax2.set_ylabel('g(r)', fontsize=12)
        ax2.set_title(f'RDF: {rdf_data["element1"]}-{rdf_data["element2"]}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 8)
        
        # 3. 扩散系数对比
        ax3 = axes[1, 0]
        elements = ['Li']
        D_values = [msd_data['D_cm2_s']]
        bars = ax3.bar(elements, D_values, color=['blue'])
        ax3.set_ylabel('Diffusion Coefficient (cm²/s)', fontsize=12)
        ax3.set_title('Ion Diffusion Coefficients', fontsize=14)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, D_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{val:.1e}', ha='center', va='bottom')
        
        # 4. 电导率
        ax4 = axes[1, 1]
        sigma = self.calculate_conductivity(msd_data['D_cm2_s'])
        sigma_val = sigma['sigma_mS_cm']
        
        # 电导率等级
        levels = ['Poor (<0.1)', 'Moderate (0.1-1)', 'Good (1-10)', 'Excellent (>10)']
        level_colors = ['red', 'orange', 'yellow', 'green']
        
        ax4.bar(['Ionic Conductivity'], [sigma_val], color='purple')
        ax4.set_ylabel('Conductivity (mS/cm)', fontsize=12)
        ax4.set_title(f'σ = {sigma_val:.2e} mS/cm', fontsize=14)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Li₃Y₀.₆₆Bi₀.₃₃Cl₆ MD Analysis @ {self.temperature}K', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n💾 图表已保存: {output_file}")
        plt.show()
    
    def export_to_csv(self, msd_data, rdf_data, output_prefix="md_analysis"):
        """导出数据到CSV"""
        # 导出MSD数据
        msd_df = pd.DataFrame({
            'Time_ps': msd_data['time'],
            'MSD_A2': msd_data['msd']
        })
        msd_df.to_csv(f"{output_prefix}_msd.csv", index=False)
        
        # 导出RDF数据
        rdf_df = pd.DataFrame({
            'Distance_A': rdf_data['distances'],
            'RDF': rdf_data['rdf']
        })
        rdf_df.to_csv(f"{output_prefix}_rdf.csv", index=False)
        
        # 导出汇总数据
        summary_df = pd.DataFrame([{
            'Temperature_K': self.temperature,
            'D_cm2_s': msd_data['D_cm2_s'],
            'D_ang2_ps': msd_data['D_ang2_ps']
        }])
        summary_df.to_csv(f"{output_prefix}_summary.csv", index=False)
        
        print(f"📁 数据已导出: {output_prefix}_*.csv")

# 主程序
def main():
    """运行MD后处理"""
    print("="*60)
    print("pymatgen MD 后处理工具")
    print("体系: Li₃Y₀.₆₆Bi₀.₃₃Cl₆ @ 700K")
    print("="*60)
    
    # 创建处理器
    processor = MDPostProcessor(trajectory_file="XDATCAR", temperature=700.0)
    
    # 加载轨迹
    if not processor.load_trajectory():
        return
    
    # 计算MSD
    msd_result = processor.calculate_msd(element="Li", time_step=1.0)
    if not msd_result:
        print("❌ MSD计算失败")
        return
    
    # 计算RDF
    rdf_result = processor.calculate_rdf(element1="Li", element2="Cl")
    
    # 计算电导率
    conductivity = processor.calculate_conductivity(msd_result['D_cm2_s'])
    
    # 绘制结果
    processor.plot_results(msd_result, rdf_result, "md_analysis_results.png")
    
    # 导出数据
    processor.export_to_csv(msd_result, rdf_result)
    
    print("\n" + "="*60)
    print("🎉 MD后处理完成!")
    print("="*60)

if __name__ == "__main__":
    main()