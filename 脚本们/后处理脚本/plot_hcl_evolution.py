"""
脚本功能：统计并可视化分子动力学轨迹中，两组原子间距离分布随时间的演化。

统计逻辑说明：
1. 数据读取：
   - 读取 VASP 的 XDATCAR 轨迹文件，获取每一帧的晶体结构。

2. 距离计算：
   - 对于轨迹中的每一帧，计算指定两组元素（如 H 和 Cl）之间所有满足截断半径（cutoff）的成对距离。
   - 距离计算考虑了晶胞的周期性边界条件 (PBC)。

3. 径向分布函数 (RDF) 修正逻辑：
   - 将距离范围 [0, max_dist] 划分为 N 个区间 (bins)。
   - 统计每一帧中落在每个区间内的原子对数量（原始直方图）。
   - 【核心几何修正】：为了消除几何体积效应（即距离越远，球壳体积越大，包含的原子自然越多的现象），
     将每个区间的原子计数除以该区间的球壳体积：
     V_shell = 4/3 * pi * (r_outer^3 - r_inner^3)
   - 结果反映了真正的“原子数密度”或径向分布函数 g(r) 的特征：
     - 峰值高表示在该距离下原子出现的概率密度远高于几何随机分布。
     - 消除了长距离处因体积增大带来的背景噪声。

4. 可视化：
   - X轴：模拟时间 (ps)。
   - Y轴：原子间距离 (Å)。
   - 颜色 (ColorBar)：修正后的概率密度/RDF强度。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Xdatcar
#统计逻辑


# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def compute_pair_distances(structure, species1, species2, cutoff=10.0):
    """
    计算两组元素之间的所有成对距离 (考虑周期性边界条件)
    """
    # 获取特定元素的索引
    indices1 = [i for i, site in enumerate(structure) if site.specie.symbol == species1]
    indices2 = [i for i, site in enumerate(structure) if site.specie.symbol == species2]
    
    if not indices1 or not indices2:
        return np.array([])

    # 获取笛卡尔坐标 (Cartesian Coords)
    # pymatgen 的 lattice.get_all_distances 需要分数坐标
    frac_coords1 = structure.frac_coords[indices1]
    frac_coords2 = structure.frac_coords[indices2]
    
    # 计算距离 (返回矩阵: len(indices1) x len(indices2))
    dists = structure.lattice.get_all_distances(frac_coords1, frac_coords2)
    
    # 展平并过滤 cutoff
    flat_dists = dists.flatten()
    return flat_dists[flat_dists <= cutoff]

def main():
    print("="*50)
    print("   H-Cl (或其他原子对) 距离分布随时间演化分析")
    print("="*50)

    # 1. 检查 XDATCAR
    xdatcar_path = "XDATCAR"
    if not os.path.exists(xdatcar_path):
        xdatcar_path = input("未找到 XDATCAR，请输入路径: ").strip()
        if not os.path.exists(xdatcar_path):
            print("❌ 文件不存在。")
            return

    # 2. 读取文件
    print(f"📂 正在读取 {xdatcar_path} (请稍候)...")
    try:
        xdatcar = Xdatcar(xdatcar_path)
        structures = xdatcar.structures
        total_frames = len(structures)
        print(f"✅ 读取成功! 总帧数: {total_frames}")
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return
    
    # 3. 参数设置
    # 时间
    try:
        potim = float(input("\n请输入 POTIM (fs) [默认 1.0]: ") or "1.0")
        nblock = int(input("请输入 NBLOCK [默认 1]: ") or "1")
    except:
        potim, nblock = 1.0, 1
    
    dt = potim * nblock
    total_time = total_frames * dt / 1000.0 # ps
    print(f"ℹ️  时间步长: {dt} fs, 总时长: {total_time:.3f} ps")

    # 元素对
    print("\n--- 设置原子对 ---")
    elem1 = input("请输入元素 1 (例如 H) [默认 H]: ").strip() or "H"
    elem2 = input("请输入元素 2 (例如 Cl) [默认 Cl]: ").strip() or "Cl"
    
    # 距离设置
    try:
        max_dist = float(input("请输入最大统计距离 (Å) [默认 6.0]: ") or "6.0")
        n_bins = int(input("请输入区间数量 (Bins) [默认 100]: ") or "100")
    except:
        max_dist = 6.0
        n_bins = 100

    # 4. 统计每一帧的分布
    print(f"\n🚀 开始计算 {elem1}-{elem2} 距离分布...")
    print(f"关注范围: 0 - {max_dist} Å")
    
    # 初始化 Histogram 矩阵: 行=距离区间, 列=时间帧
    # bins edges
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # --- 💡 预计算球壳体积 (Shell Volume) ---
    # V_shell = 4/3 * pi * (r_outer^3 - r_inner^3)
    # 用于归一化，消除 r^2 的几何增长效应，得到真正的 g(r) 径向分布特性
    shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

    # 结果矩阵
    density_map = np.zeros((n_bins, total_frames))
    
    for i, struct in enumerate(structures):
        # 显示进度
        if  (i+1) % 10 == 0:
            sys.stdout.write(f"\r进度: {i+1}/{total_frames}")
            sys.stdout.flush()
            
        dists = compute_pair_distances(struct, elem1, elem2, cutoff=max_dist)
        
        if len(dists) > 0:
            # 1. 计算原始计数直方图 (不使用 density=True，我们要手动归一化)
            hist, _ = np.histogram(dists, bins=bin_edges)
            
            # 2. RDF 归一化 (除以球壳体积)
            # 这样处理后，峰值强度不再随距离增加而自然变大，更有物理意义
            rdf_hist = hist / shell_volumes
            
            # 3. (可选) 全局密度归一化，使其趋近于 1 (这里为了热力图对比，暂只做体积修正)
            # 简单的体积修正通常足够显示相对强弱
            
            density_map[:, i] = rdf_hist
        else:
            density_map[:, i] = 0

    print("\n✅ 计算完成！开始绘图...")

    # 5. 绘图 (Heatmap)
    plt.figure(figsize=(10, 6))
    
    # 构造网格
    # X轴: 时间 (ps)
    time_array = np.arange(total_frames) * dt / 1000.0 # ps
    
    # 使用 pcolormesh 绘制
    X, Y = np.meshgrid(time_array, bin_centers)
    
    # 绘制热力图
    # 💡 使用 'jet' 或 'viridis'，并设置 vmax 避免极高值掩盖细节
    # RDF 在短距离(r->0)可能因为体积趋近0导致数值极大，可以截断 max 值
    #current_max = np.percentile(density_map, 99) # 取 99 分位数作为颜色上限，防止奇点
    # 降低这个值，可以让弱信号（比如次近邻层）更明显
    current_max = np.percentile(density_map, 95) # 尝试把 99 改成 90 或 95
    plt.pcolormesh(X, Y, density_map, cmap='jet', shading='auto', vmin=0, vmax=current_max)
    
    cbar = plt.colorbar()
    cbar.set_label('径向分布函数 g(r) (Arbitrary Units)', fontsize=12)
    
    plt.xlabel("时间 (ps)", fontsize=12)
    plt.ylabel(f"{elem1}-{elem2} 距离 ($\AA$)", fontsize=12)
    plt.title(f"{elem1}-{elem2} 距离分布随时间演化图", fontsize=14)
    
    plt.ylim(0, max_dist)
    plt.xlim(0, total_time)
    
    # 保存
    filename = f"dist_evolution_{elem1}_{elem2}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"🎉 图片已保存为: {filename}")
    plt.show()

if __name__ == "__main__":
    main()
