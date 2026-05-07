#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ActivationEnergyAnalyzer:
    """活化能分析器 - 基于阿伦尼乌斯公式"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.temperatures = []  # 温度 (K)
        self.conductivities = []  # 电导率 (mS/cm)
        self.results = []
        
    def find_result_files(self, pattern="*K"):
        """查找包含result.dat的文件夹"""
        print(f"🔍 在 {self.base_dir} 中查找结果文件...")
        
        temperature_dirs = []
        
        # 查找所有匹配 pattern 的文件夹
        for item in self.base_dir.glob(pattern):
            if item.is_dir():
                result_file = item / "result.dat"
                if result_file.exists():
                    temperature_dirs.append((str(item), result_file))
                    print(f"  ✅ 找到: {item}")
        
        return temperature_dirs
    
    def extract_temperature_from_dir(self, dir_name):
        """从文件夹名中提取温度值"""
        dir_name = str(dir_name)
        dir_name = os.path.basename(dir_name)
        
        import re
        match = re.search(r'(\d+\.?\d*)', dir_name)
        if match:
            return float(match.group(1))
        
        return None
    
    def parse_result_file(self, filepath):
        """解析result.dat文件"""
        temp = None
        conductivity = None
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        c = float(parts[1])
                        temp = t
                        conductivity = c
                        break
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"❌ 解析文件 {filepath} 时出错: {e}")
        
        return temp, conductivity
    
    def collect_data(self, pattern="*K"):
        """收集所有温度的电导率数据"""
        dirs_and_files = self.find_result_files(pattern)
        
        for dir_path, file_path in dirs_and_files:
            temp_from_dir = self.extract_temperature_from_dir(dir_path)
            temp_from_file, conductivity = self.parse_result_file(file_path)
            
            if temp_from_file is not None:
                temperature = temp_from_file
            elif temp_from_dir is not None:
                temperature = temp_from_dir
            else:
                print(f"⚠️  无法从 {dir_path} 确定温度，跳过")
                continue
            
            if conductivity is None:
                print(f"⚠️  无法从 {file_path} 读取电导率，跳过")
                continue
            
            self.temperatures.append(temperature)
            self.conductivities.append(conductivity)
            self.results.append({
                'temperature': temperature,
                'conductivity': conductivity,
                'directory': dir_path,
                'file': str(file_path)
            })
            
            print(f"  📊 {temperature}K: {conductivity:.2e} mS/cm")
    
    def calculate_activation_energy_manual(self):
        """手动计算活化能（根据提供的公式）"""
        if len(self.temperatures) < 2:
            print("❌ 需要至少2个温度点的数据")
            return None
        
        # 准备数据
        T = np.array(self.temperatures)  # 温度 (K)
        sigma = np.array(self.conductivities)  # 电导率 (S/cm)
        
        print(f"\n{'='*60}")
        print("手动计算活化能（阿伦尼乌斯公式）")
        print('='*60)
        
        # 步骤1: 计算预处理参数
        print("\n📊 数据预处理:")
        print(f"{'T(K)':<10} {'σ(mS/cm)':<15} {'1/T (K⁻¹)':<15} {'σ×T':<20} {'ln(σT)':<15}")
        print("-"*80)
        
        inv_T_list = []
        sigmaT_list = []
        ln_sigmaT_list = []
        
        for i in range(len(T)):
            inv_T = 1.0 / T[i]  # 1/T
            sigmaT = sigma[i] * T[i]  # σ×T
            ln_sigmaT = np.log(sigmaT)  # ln(σT)
            
            inv_T_list.append(inv_T)
            sigmaT_list.append(sigmaT)
            ln_sigmaT_list.append(ln_sigmaT)
            
            print(f"{T[i]:<10.1f} {sigma[i]:<15.2e} {inv_T:<15.6f} {sigmaT:<20.2e} {ln_sigmaT:<15.6f}")
        
        inv_T = np.array(inv_T_list)
        sigmaT = np.array(sigmaT_list)
        ln_sigmaT = np.array(ln_sigmaT_list)
        
        # 步骤2: 计算线性回归所需的和
        n = len(T)
        sum_x = np.sum(inv_T)
        sum_y = np.sum(ln_sigmaT)
        sum_xy = np.sum(inv_T * ln_sigmaT)
        sum_x2 = np.sum(inv_T**2)
        sum_y2 = np.sum(ln_sigmaT**2)
        
        print(f"\n📈 统计求和:")
        print(f"  数据点数 n = {n}")
        print(f"  Σ(1/T) = {sum_x:.6f}")
        print(f"  Σln(σT) = {sum_y:.6f}")
        print(f"  Σ[(1/T)×ln(σT)] = {sum_xy:.6f}")
        print(f"  Σ(1/T)² = {sum_x2:.6e}")
        
        # 步骤3: 计算斜率和截距
        denominator = n * sum_x2 - sum_x**2
        if abs(denominator) < 1e-10:
            print("❌ 分母为0，无法计算斜率")
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        print(f"\n📈 线性回归参数:")
        print(f"  分母 = {denominator:.6e}")
        print(f"  斜率 k = {slope:.6f}")
        print(f"  截距 = {intercept:.6f}")
        
        # 步骤4: 计算相关系数
        numerator_r = n * sum_xy - sum_x * sum_y
        denominator_r = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        r_value = numerator_r / denominator_r if denominator_r != 0 else 0
        
        print(f"\n📈 相关性分析:")
        print(f"  相关系数 R = {r_value:.6f}")
        print(f"  确定系数 R² = {r_value**2:.6f}")
        
        # 步骤5: 计算活化能
        R = 8.314462618  # 气体常数，单位: J/(mol·K)
        R_eV = 8.617333262e-5  # 气体常数，单位: eV/K
        
        # 根据公式: 斜率 k = -Ea/R
        Ea_J = -slope * R  # 单位: J/mol
        Ea_kJ = Ea_J / 1000.0  # 单位: kJ/mol
        Ea_eV = -slope * R_eV  # 单位: eV
        
        print(f"\n⚡ 活化能计算:")
        print(f"  气体常数 R = {R:.6f} J/(mol·K)")
        print(f"  气体常数 R = {R_eV:.6e} eV/K")
        print(f"  活化能 Ea = -k × R = {-slope:.6f} × {R:.6f}")
        print(f"  Ea = {Ea_kJ:.4f} kJ/mol")
        print(f"  Ea = {Ea_eV:.6f} eV")
        
        # 步骤6: 计算指前因子
        A = np.exp(intercept)  # 指前因子
        print(f"\n📊 指前因子:")
        print(f"  lnA = 截距 = {intercept:.6f}")
        print(f"  A = exp(截距) = {A:.6e} mS·K/cm")
        
        return {
            'temperatures': T,
            'conductivities': sigma,
            'inv_T': inv_T,
            'sigmaT': sigmaT,
            'ln_sigmaT': ln_sigmaT,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'Ea_kJ_mol': Ea_kJ,
            'Ea_eV': Ea_eV,
            'Ea_J_mol': Ea_J,
            'prefactor': A,
            'n_points': n,
            'sum_x': sum_x,
            'sum_y': sum_y,
            'sum_xy': sum_xy,
            'sum_x2': sum_x2
        }
    
    def calculate_activation_energy_scipy(self):
        """使用scipy进行线性回归"""
        if len(self.temperatures) < 2:
            print("❌ 需要至少2个温度点的数据")
            return None
        
        T = np.array(self.temperatures)
        sigma = np.array(self.conductivities)
        
        # 计算预处理参数
        inv_T = 1.0 / T
        sigmaT = sigma * T
        ln_sigmaT = np.log(sigmaT)
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_sigmaT)
        
        # 计算活化能
        R = 8.314462618
        R_eV = 8.617333262e-5
        
        Ea_kJ = -slope * R / 1000.0
        Ea_eV = -slope * R_eV
        
        A = np.exp(intercept)
        
        return {
            'temperatures': T,
            'conductivities': sigma,
            'inv_T': inv_T,
            'sigmaT': sigmaT,
            'ln_sigmaT': ln_sigmaT,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'Ea_kJ_mol': Ea_kJ,
            'Ea_eV': Ea_eV,
            'prefactor': A
        }
    
    def plot_arrhenius(self, fit_results, method="manual", output_dir="."):
        """绘制阿伦尼乌斯图"""
        if fit_results is None:
            return
        
        T = fit_results['temperatures']
        sigma = fit_results['conductivities']
        inv_T = fit_results['inv_T']
        ln_sigmaT = fit_results['ln_sigmaT']
        slope = fit_results['slope']
        intercept = fit_results['intercept']
        Ea_kJ = fit_results['Ea_kJ_mol']
        Ea_eV = fit_results['Ea_eV']
        r2 = fit_results.get('r_squared', 0)
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 子图1: 阿伦尼乌斯图 (ln(σT) vs 1/T)
        ax1 = axes[0]
       # ax1.scatter(inv_T, ln_sigmaT, s=100, color='blue', edgecolor='black', zorder=5, label='实验数据')
        ax1.scatter(inv_T, ln_sigmaT, s=100, color='blue', edgecolor='black',
                   zorder=5)
        
        # 拟合线
        fit_line = intercept + slope * inv_T
        ax1.plot(inv_T, fit_line, 'r-', linewidth=2, 
                label=f'\nEa = {Ea_kJ:.3f} kJ/mol\nR² = {r2:.4f}')
        
        ax1.set_xlabel('1/T (K⁻¹)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ln(σT)', fontsize=12, fontweight='bold')
        ax1.set_title('ln(σT) vs 1/T', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best')
        
        # 添加数据标签
        for i, (x, y, t) in enumerate(zip(inv_T, ln_sigmaT, T)):
            ax1.annotate(f'{t:.0f}K', xy=(x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, fontweight='bold')
        
        # 子图2: σT vs 1/T (半对数坐标)
        ax2 = axes[1]
        ax2.scatter(inv_T, sigma * T, s=100, color='green', edgecolor='black', 
                   zorder=5, label='实验数据')
        
        # 拟合曲线
        sigmaT_fit = np.exp(intercept + slope * inv_T)
        ax2.plot(inv_T, sigmaT_fit, 'r-', linewidth=2, 
                label=f'阿伦尼乌斯拟合\nEa = {Ea_eV:.3f} eV')
        
        ax2.set_xlabel('1/T (K⁻¹)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('σT (mS·K/cm)', fontsize=12, fontweight='bold')
        ax2.set_title('阿伦尼乌斯关系: σT vs 1/T', fontsize=13, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--', which='both')
        ax2.legend(loc='best')
        
        # 子图3: 电导率随温度变化
        ax3 = axes[2]
        sort_idx = np.argsort(T)
        T_sorted = T[sort_idx]
        sigma_sorted = sigma[sort_idx]
        
        ax3.plot(T_sorted, sigma_sorted, 'bo-', linewidth=2, markersize=8, 
                label='实验数据')
        
        # 拟合曲线
        T_fit = np.linspace(min(T)*0.9, max(T)*1.1, 100)
        A = np.exp(intercept)
        sigma_fit = A * np.exp(-Ea_eV/(8.617e-5 * T_fit)) / T_fit
        
        ax3.plot(T_fit, sigma_fit, 'r--', linewidth=2, 
                label=f'拟合曲线\nEa = {Ea_kJ:.2f} kJ/mol')
        
        ax3.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Ionic Conductivity (mS/cm)', fontsize=12, fontweight='bold')
        ax3.set_title('电导率温度依赖关系', fontsize=13, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, linestyle='--', which='both')
        ax3.legend(loc='best')
        
        plt.suptitle(f'离子电导率阿伦尼乌斯分析 (方法: {method})\n活化能 Ea = {Ea_kJ:.3f} kJ/mol ({Ea_eV:.4f} eV)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图片
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"activation_energy_{method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📈 图表已保存: {output_file}")
        
        plt.show()
        return output_file
    
    def save_detailed_results(self, fit_results, method="manual", output_dir="."):
        """保存详细的拟合结果"""
        if fit_results is None:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        result_file = output_dir / f"activation_energy_{method}_detailed.txt"
        
        with open(result_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("离子电导率活化能分析 - 详细计算过程\n")
            f.write(f"计算方法: {method}\n")
            f.write("="*80 + "\n\n")
            
            f.write("一、原始数据\n")
            f.write("-"*80 + "\n")
            f.write(f"{'温度 T(K)':<12} {'电导率 σ(mS/cm)':<20} {'1/T (K⁻¹)':<15} {'σ×T':<20} {'ln(σT)':<15}\n")
            f.write("-"*80 + "\n")
            
            for i in range(len(fit_results['temperatures'])):
                f.write(f"{fit_results['temperatures'][i]:<12.1f} "
                       f"{fit_results['conductivities'][i]:<20.2e} "
                       f"{fit_results['inv_T'][i]:<15.6f} "
                       f"{fit_results['sigmaT'][i]:<20.2e} "
                       f"{fit_results['ln_sigmaT'][i]:<15.6f}\n")
            
            f.write("\n\n二、统计求和\n")
            f.write("-"*80 + "\n")
            f.write(f"数据点数 n = {fit_results.get('n_points', len(fit_results['temperatures']))}\n")
            f.write(f"Σ(1/T) = {fit_results.get('sum_x', 0):.6f}\n")
            f.write(f"Σln(σT) = {fit_results.get('sum_y', 0):.6f}\n")
            f.write(f"Σ[(1/T)×ln(σT)] = {fit_results.get('sum_xy', 0):.6f}\n")
            f.write(f"Σ(1/T)² = {fit_results.get('sum_x2', 0):.6e}\n")
            
            f.write("\n\n三、线性回归结果\n")
            f.write("-"*80 + "\n")
            f.write(f"斜率 k = {fit_results['slope']:.6f}\n")
            f.write(f"截距 = {fit_results['intercept']:.6f}\n")
            f.write(f"相关系数 R = {fit_results.get('r_value', 0):.6f}\n")
            f.write(f"确定系数 R² = {fit_results.get('r_squared', 0):.6f}\n")
            if 'std_err' in fit_results:
                f.write(f"标准误差 = {fit_results['std_err']:.6f}\n")
            if 'p_value' in fit_results:
                f.write(f"p值 = {fit_results['p_value']:.6e}\n")
            
            f.write("\n\n四、活化能计算结果\n")
            f.write("-"*80 + "\n")
            f.write(f"气体常数 R = 8.314462618 J/(mol·K)\n")
            f.write(f"气体常数 R = 8.617333262e-5 eV/K\n")
            f.write(f"根据公式: Ea = -k × R\n")
            f.write(f"活化能 Ea = {fit_results['Ea_kJ_mol']:.6f} kJ/mol\n")
            f.write(f"活化能 Ea = {fit_results['Ea_eV']:.6f} eV\n")
            
            f.write("\n\n五、指前因子\n")
            f.write("-"*80 + "\n")
            f.write(f"lnA = 截距 = {fit_results['intercept']:.6f}\n")
            f.write(f"A = exp(截距) = {fit_results['prefactor']:.6e} mS·K/cm\n")
            
            f.write("\n\n六、阿伦尼乌斯方程\n")
            f.write("-"*80 + "\n")
            f.write(f"σT = A × exp(-Ea/RT)\n")
            f.write(f"σT = {fit_results['prefactor']:.3e} × exp(-{fit_results['Ea_kJ_mol']:.3f}/(RT))\n")
            f.write(f"其中 R = 8.314 J/(mol·K)\n")
        
        print(f"📁 详细结果已保存: {result_file}")
        return result_file
    
    def save_summary_csv(self, fit_results_manual, fit_results_scipy, output_dir="."):
        """保存汇总结果到CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        summary_data = []
        
        if fit_results_manual:
            summary_data.append({
                'Method': 'Manual Calculation',
                'Ea_kJ_mol': fit_results_manual['Ea_kJ_mol'],
                'Ea_eV': fit_results_manual['Ea_eV'],
                'Slope': fit_results_manual['slope'],
                'Intercept': fit_results_manual['intercept'],
                'R_squared': fit_results_manual.get('r_squared', 0),
                'Prefactor': fit_results_manual['prefactor'],
                'n_points': fit_results_manual.get('n_points', len(fit_results_manual['temperatures']))
            })
        
        if fit_results_scipy:
            summary_data.append({
                'Method': 'Scipy Linear Regression',
                'Ea_kJ_mol': fit_results_scipy['Ea_kJ_mol'],
                'Ea_eV': fit_results_scipy['Ea_eV'],
                'Slope': fit_results_scipy['slope'],
                'Intercept': fit_results_scipy['intercept'],
                'R_squared': fit_results_scipy['r_squared'],
                'Prefactor': fit_results_scipy['prefactor'],
                'n_points': len(fit_results_scipy['temperatures'])
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = output_dir / "activation_energy_summary.csv"
            df.to_csv(csv_file, index=False, float_format='%.6f')
            print(f"📁 汇总表格已保存: {csv_file}")
            return csv_file
        
        return None
   
    def run_analysis(self, pattern="*K", output_dir="./activation_energy_analysis"):
        """运行完整分析流程"""
        print("="*80)
        print("离子电导率活化能分析 - 阿伦尼乌斯公式拟合")
        print("="*80)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # 收集数据
        self.collect_data(pattern)
        
        if len(self.temperatures) < 2:
            print("❌ 数据不足，需要至少2个温度点")
            return
        
        print(f"\n📊 共收集到 {len(self.temperatures)} 个温度点的数据")
        
        # 按温度排序
        sort_idx = np.argsort(self.temperatures)
        self.temperatures = np.array(self.temperatures)[sort_idx]
        self.conductivities = np.array(self.conductivities)[sort_idx]
        
        # 重新组织results
        sorted_results = []
        for idx in sort_idx:
            sorted_results.append(self.results[idx])
        self.results = sorted_results
        
        # 方法1: 手动计算
        print(f"\n{'='*60}")
        print("方法1: 手动计算（基于您提供的公式）")
        print('='*60)
        fit_results_manual = self.calculate_activation_energy_manual()
        
        # 方法2: 使用scipy计算
        print(f"\n{'='*60}")
        print("方法2: 使用Scipy线性回归")
        print('='*60)
        fit_results_scipy = self.calculate_activation_energy_scipy()
        
        if fit_results_scipy:
            print(f"  📈 线性回归结果:")
            print(f"    斜率: {fit_results_scipy['slope']:.6f}")
            print(f"    截距: {fit_results_scipy['intercept']:.6f}")
            print(f"    R²值: {fit_results_scipy['r_squared']:.6f}")
            print(f"    活化能 Ea: {fit_results_scipy['Ea_kJ_mol']:.4f} kJ/mol")
            print(f"    活化能 Ea: {fit_results_scipy['Ea_eV']:.6f} eV")
            print(f"    指前因子 A: {fit_results_scipy['prefactor']:.4e} S·K/cm")
        
        # 比较两种方法的结果
        if fit_results_manual and fit_results_scipy:
            print(f"\n{'='*60}")
            print("结果比较")
            print('='*60)
            print(f"  {'参数':<20} {'手动计算':<15} {'Scipy回归':<15} {'差异':<10}")
            print(f"  {'-'*20:<20} {'-'*15:<15} {'-'*15:<15} {'-'*10:<10}")
            
            diff_Ea = abs(fit_results_manual['Ea_kJ_mol'] - fit_results_scipy['Ea_kJ_mol'])
            print(f"  {'Ea (kJ/mol)':<20} {fit_results_manual['Ea_kJ_mol']:<15.4f} "
                  f"{fit_results_scipy['Ea_kJ_mol']:<15.4f} {diff_Ea:<10.4f}")
            
            diff_slope = abs(fit_results_manual['slope'] - fit_results_scipy['slope'])
            print(f"  {'斜率':<20} {fit_results_manual['slope']:<15.6f} "
                  f"{fit_results_scipy['slope']:<15.6f} {diff_slope:<10.6f}")
            
            diff_r2 = abs(fit_results_manual.get('r_squared', 0) - fit_results_scipy['r_squared'])
            print(f"  {'R²':<20} {fit_results_manual.get('r_squared', 0):<15.6f} "
                  f"{fit_results_scipy['r_squared']:<15.6f} {diff_r2:<10.6f}")
        
        # 保存原始数据
        data_df = pd.DataFrame(self.results)
        data_file = Path(output_dir) / "conductivity_raw_data.csv"
        data_df.to_csv(data_file, index=False)
        print(f"\n📁 原始数据已保存: {data_file}")
        
        # 绘制图表
        if fit_results_manual:
            self.plot_arrhenius(fit_results_manual, "manual", output_dir)
            self.save_detailed_results(fit_results_manual, "manual", output_dir)
        
        if fit_results_scipy:
            self.plot_arrhenius(fit_results_scipy, "scipy", output_dir)
            self.save_detailed_results(fit_results_scipy, "scipy", output_dir)
        
        # 保存汇总结果
        self.save_summary_csv(fit_results_manual, fit_results_scipy, output_dir)
        
        print(f"\n✅ 分析完成！所有结果保存在: {output_dir}/")
        
        return {
            'manual': fit_results_manual,
            'scipy': fit_results_scipy
        }

"""
# 示例使用
if __name__ == "__main__":
    # 创建分析器
    analyzer = ActivationEnergyAnalyzer(base_dir=".")
    
    # 运行分析
    results = analyzer.run_analysis(
        pattern="*K",  # 匹配包含"K"的文件夹
        output_dir="./activation_energy_analysis"
        )
"""
if __name__ == "__main__":
    # 创建分析器，指定你的 base 路径
    analyzer = ActivationEnergyAnalyzer(base_dir=r"D:\tbb-LPSCI\AIMD\base")
    
    # 运行分析（它会自动寻找匹配 *K 的子文件夹）
    results = analyzer.run_analysis(
        pattern="*K",
        output_dir="./activation_energy_analysis"
    )

'''    def save_results(self, fit_results, output_dir="."):
        """保存分析结果"""
        if fit_results is None:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
'''






