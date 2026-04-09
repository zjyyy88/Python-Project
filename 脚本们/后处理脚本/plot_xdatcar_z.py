import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Xdatcar
from scipy.stats import gaussian_kde

# 尝试设置中文字体，如果失败则回退到英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def get_smoothed_density(z_coords, c_length, sigma=0.2, n_points=500):
    """
    使用高斯展宽计算密度分布
    :param z_coords: 原子 Z 坐标列表
    :param c_length: 晶胞 C 轴长度
    :param sigma: 高斯展宽的标准差 (Angstrom)
    :param n_points: 采样点数
    :return: z_grid, density
    """
    z_grid = np.linspace(0, c_length, n_points)
    density = np.zeros_like(z_grid)
    
    # 对每个原子添加高斯贡献
    # 为了效率，可以只计算附近的点，但对于一维数组，直接全算也可以
    # 考虑到周期性边界条件(PBC)？这里假设Z轴分布包含真空层或者关注特定区域，
    # 简化起见先不处理跨越边界的平滑（除非原子正好在0或c附近）
    # 简单的 PBC 处理：如果原子在 0 附近，也在 c 附近贡献
    
    normalization = 1 / (sigma * np.sqrt(2 * np.pi))
    
    for z0 in z_coords:
        # 主像
        dist = z_grid - z0
        density += normalization * np.exp(-0.5 * (dist / sigma) ** 2)
        
        # 简单的 PBC 镜像 (考虑 +/- c)
        dist_plus = z_grid - (z0 + c_length)
        density += normalization * np.exp(-0.5 * (dist_plus / sigma) ** 2)
        
        dist_minus = z_grid - (z0 - c_length)
        density += normalization * np.exp(-0.5 * (dist_minus / sigma) ** 2)
        
    return z_grid, density

def main():
    print("="*40)
    print("      XDATCAR 原子 Z 轴分布分析工具")
    print("="*40)
    
    xdatcar_path = "XDATCAR"
    if not os.path.exists(xdatcar_path):
        xdatcar_path = input("未找到 XDATCAR，请输入文件路径: ").strip()
        if not os.path.exists(xdatcar_path):
            print("❌ 错误: 文件不存在。")
            return

    print(f"📂 正在读取 {xdatcar_path} (可能需要几秒钟)...")
    try:
        xdatcar = Xdatcar(xdatcar_path)
        structures = xdatcar.structures
        total_frames = len(structures)
        print(f"✅ 读取成功! 总帧数: {total_frames}")
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return

    # 获取时间参数
    print("\n⏱️  时间设置 (用于将时间转换为帧数)")
    try:
        potim = float(input("请输入 POTIM (时间步长, fs) [默认 1.0]: ") or "1.0")
        nblock = int(input("请输入 NBLOCK (写入频率) [默认 1]: ") or "1")
    except ValueError:
        print("❌ 输入无效，使用默认值。")
        potim = 1.0
        nblock = 1
    
    time_per_frame = potim * nblock # fs
    total_time_ps = total_frames * time_per_frame / 1000.0
    print(f"ℹ️  每帧对应时间: {time_per_frame} fs")
    print(f"ℹ️  总模拟时间: {total_time_ps:.3f} ps")

    while True:
        print("-" * 30)
        user_input = input(f"请输入在哪个时间点绘图 (fs) (范围 0 - {total_frames * time_per_frame}): \n(或者输入 'q' 退出): ").strip()
        
        if user_input.lower() == 'q':
            break
            
        try:
            target_time = float(user_input)
            # 计算最近的帧
            frame_idx = int(round(target_time / time_per_frame)) - 1
            
            # 边界检查
            if frame_idx < 0: frame_idx = 0
            if frame_idx >= total_frames: 
                print(f"⚠️ 时间超出范围，将使用最后一帧 (Frame {total_frames})")
                frame_idx = total_frames - 1
            
            actual_time = (frame_idx + 1) * time_per_frame
            print(f"🔎 目标时间: {target_time} fs -> 对应帧: {frame_idx + 1} (实际时间 {actual_time} fs)")
            
            # 获取结构数据
            struct = structures[frame_idx]
            lattice = struct.lattice
            c_length = lattice.c
            
            # 准备绘图
            plt.figure(figsize=(10, 6))
            
            # 收集原子坐标
            # elements = struct.composition.element_composition.keys() # 默认：所有元素

            # --- 自定义元素设置 ---
            # 💡💡💡💡💡💡💡💡💡💡请在这里输入您想绘制的元素符号，例如 ["Bi", "I", "Cs"]
            # 如果列表为空 []，则默认绘制所有元素
            #target_elements_str = ["Li", "In", "Cl"] 
            target_elements_str = ["O", "H", "Cl"] 
            # 💡💡💡💡💡💡💡💡💡💡请在这里输入您想绘制的元素符号

            all_elements_in_struct = list(struct.composition.element_composition.keys())
            
            if target_elements_str:
                # 过滤：只保留结构中存在的元素
                elements = [e for e in all_elements_in_struct if e.symbol in target_elements_str]
                if not elements:
                    print(f"⚠️ 警告: 在结构中未找到您指定的元素 {target_elements_str}。将绘制所有元素。")
                    elements = all_elements_in_struct
            else:
                 elements = all_elements_in_struct
            
            processed_elements = [] 
            
            # 获取所有原子 Z 坐标 (Cartesian)
            all_sites = struct.sites
            
            # 使用高斯平滑还是直方图？
            plot_type = "smooth"  # 设置为 "hist" 画条形图，设置为 "smooth" 画平滑曲线
            
            # --- 以前的默认颜色配置 (已注释) ---
            # colors = plt.cm.tab10(np.linspace(0, 1, len(elements)))

            # --- 当前使用的自定义颜色配置 ---
            # 💡💡💡💡💡💡💡💡💡💡 在这里修改特定元素的颜色
            custom_colors = {
                "Li": "purple",   # 锂 -> 紫色
                "Y": "orange", 
                "In": "orange",  # 铟 -> 橙色
                "Cl": "green",   # 氯 -> 绿色
                "H": "#1f77b4",  # 支持十六进制颜色
                "O": "red"       
            }
            # 💡💡💡💡💡💡💡💡💡💡 在这里修改特定元素的颜色

            # 自动分配颜色 (作为默认备选，处理未自定义的元素)
            default_colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(elements))))
            
            for i, elem in enumerate(elements):
                # 确定当前元素的颜色
                sym = elem.symbol
                if sym in custom_colors:
                    cur_color = custom_colors[sym]
                else:
                    cur_color = default_colors[i % len(default_colors)]

                # 筛选该元素的原子
                elem_sites = [site for site in all_sites if site.specie == elem]
                z_coords = [site.coords[2] for site in elem_sites]
                
                if not z_coords:
                    continue
                
                label_txt = f"{elem.symbol} ({len(z_coords)} atoms)"
                
                if plot_type == "smooth":
                    # --- 纵轴是密度 (Density) ---
                    # 曲线下面积 = 总原子数
                    z_grid, density = get_smoothed_density(z_coords, c_length, sigma=0.1)
                    plt.plot(z_grid, density, label=label_txt, linewidth=2, color=cur_color)
                    plt.fill_between(z_grid, density, alpha=0.2, color=cur_color)
                    plt.ylabel("原子数密度 (Atoms/Å)", fontsize=12) # 更新Y轴标签
                    
                else:
                    # --- 纵轴是原子个数 (Count) ---
                    # bins=100 表示把 z 轴切成 100 份
                    plt.hist(z_coords, bins=100, range=(0, c_length), alpha=0.6, 
                             label=label_txt, color=cur_color, edgecolor='None')
                    plt.ylabel("原子个数 (Count)", fontsize=12) # 更新Y轴标签

            plt.xlabel("Z 轴坐标 ($\AA$)", fontsize=12)
            # plt.ylabel 已经在上面设置了
            plt.title(f"原子 Z 轴分布 @ {actual_time} fs (Frame {frame_idx+1})", fontsize=14)
            plt.xlim(0, c_length)
            
            # --- 💡💡💡💡💡💡💡💡💡💡💡 固定纵轴范围 (可选) ---
            # 如果想让不同时间的图纵轴保持一致，请取消下面这行的注释并修改数值
            plt.ylim(0, 35)  # 例如: 纵轴固定在 0 到 15
            
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 标记晶胞边界
            plt.axvline(x=0, color='k', linestyle='-', linewidth=1)
            plt.axvline(x=c_length, color='k', linestyle='-', linewidth=1)
            
            output_filename = f"z_distribution_{int(target_time)}fs.png"
            plt.savefig(output_filename, dpi=300)
            print(f"✅ 图片已保存: {output_filename}")
            
            # 尝试显示图片
            try:
                plt.show()
            except:
                pass
                
        except ValueError:
            print("❌ 输入无效，请输入数字。")
            
if __name__ == "__main__":
    main()
