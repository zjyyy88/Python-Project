"""
脚本功能：计算并绘制标准径向分布函数 (RDF, g(r)) 随时间的演化热力图。

RDF g(r) 定义说明：
g(r) 表示在距离参考粒子 r 处找到另一个粒子的概率，相对于理想气体（均匀分布）的概率的比值。

计算公式：
           dN(r)
g(r) = ---------------
       4πr²dr * ρ_bulk

其中：
- dN(r): 在距离 r 到 r+dr 的球壳内找到的粒子数。
- 4πr²dr: 该球壳的体积。
- ρ_bulk: 体系的平均数密度 (N_total / V_cell)。

物理意义：
- g(r) > 1 : 表示粒子在该位置富集（如成键、配位层）。
- g(r) < 1 : 表示粒子在该位置贫化（如原子核排斥体积）。
- g(r) -> 1: 当 r 很大时，表示关联消失，趋于均匀分布。

本脚本特点：
1. 标准化归一化：结果无量纲，背景值趋于 1。
2. 考虑 N-N (同种) 和 N-M (异种) 原子对计算时的密度差异。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp import Xdatcar

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def compute_pair_distances(structure, species1, species2, cutoff=10.0):
    """
    计算两组元素之间的所有成对距离
    """
    indices1 = [i for i, site in enumerate(structure) if site.specie.symbol == species1]
    indices2 = [i for i, site in enumerate(structure) if site.specie.symbol == species2]
    
    # 获取原子数量，用于后续计算全局密度
    n1 = len(indices1)
    n2 = len(indices2)
    
    if n1 == 0 or n2 == 0:
        return np.array([]), 0, 0

    frac_coords1 = structure.frac_coords[indices1]
    frac_coords2 = structure.frac_coords[indices2]
    
    dists = structure.lattice.get_all_distances(frac_coords1, frac_coords2)
    
    # 如果是同种原子 (e.g., Li-Li)，get_all_distances 会包含对角线上的 0 (自己到自己的距离)
    # 必须把这些 0 剔除，因为它们不是成对相互作用
    if species1 == species2:
        # 剔除距离极小的值 (浮点数误差考虑，小于 0.01 Å 视为自相互作用)
        flat_dists = dists[dists > 0.01].flatten()
    else:
        flat_dists = dists.flatten()
        
    return flat_dists[flat_dists <= cutoff], n1, n2

def main():
    print("="*60)
    print("   标准均质化 RDF g(r) 随时间演化分析工具 (Standardized)")
    print("="*60)

    # 1. 读取 XDATCAR
    xdatcar_path = "XDATCAR"
    if not os.path.exists(xdatcar_path):
        xdatcar_path = input("未找到 XDATCAR，请输入路径: ").strip()
    
    if not os.path.exists(xdatcar_path):
        print("❌ 文件不存在。")
        return

    print(f"📂 正在读取 {xdatcar_path} ...")
    try:
        xdatcar = Xdatcar(xdatcar_path)
        structures = xdatcar.structures
        total_frames = len(structures)
        print(f"✅ 读取成功! 总帧数: {total_frames}")
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return
    
    # 2. 参数输入
    try:
        potim = float(input("\n请输入 POTIM (fs) [默认 1.0]: ") or "1.0")
        nblock = int(input("请输入 NBLOCK [默认 1]: ") or "1")
        dt = potim * nblock
        total_time_ps = total_frames * dt / 1000.0
    except:
        dt, total_time_ps = 1.0, total_frames/1000.0

    print("\n--- 设置原子对 ---")
    elem1 = input("请输入中心原子 (Ref, e.g. Li) [默认 Li]: ").strip() or "Li"
    elem2 = input("请输入配位原子 (Target, e.g. Cl) [默认 Cl]: ").strip() or "Cl"
    
    try:
        max_dist = float(input("最大统计距离 (Å) [默认 8.0]: ") or "8.0")
        n_bins = int(input("区间数量 (bins) [默认 100]: ") or "100")
    except:
        max_dist, n_bins = 8.0, 100

    print(f"\n🚀 开始计算 标准 RDF g(r)... (Ref:{elem1} -> Target:{elem2})")

    # 3. 准备 Histogram 参数
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dr = bin_edges[1] - bin_edges[0] # 区间宽度
    
    # 预计算每一层的球壳体积 dV = 4*pi*r^2*dr
    # 更精确的方法是用体积差 V(r+dr) - V(r)
    shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    rdf_map = np.zeros((n_bins, total_frames))
    
    for i, struct in enumerate(structures):
        if (i+1) % 10 == 0:
            sys.stdout.write(f"\r进度: {i+1}/{total_frames}")
            sys.stdout.flush()

        # 计算距离
        dists, n_ref, n_target = compute_pair_distances(struct, elem1, elem2, cutoff=max_dist)
        
        if len(dists) == 0:
            continue
            
        # --- 核心归一化逻辑 ---
        
        # 1. 体系体积 V
        vol = struct.volume
        
        # 2. 平均数密度 rho_bulk
        # 这里的定义很关键：
        # 对于 g_AB(r)，我们是站在 A 原子的角度看 B 原子的分布。
        # 理论上，如果 B 均匀分布，密度就是 N_B / Volume。
        # 但是，我们要计算的是“每个 A 原子”周围的情况，所以总观测次数是 n_ref * n_target（或 n_ref * n_pairs）
        
        # 使用 pymatgen 的 dists 数组长度已经是所有 Ref 到所有 Target 的距离总数
        # 理想情况下（均匀分布），在体积 V 内有 n_target 个目标原子
        # 所以对于某 *一个* Ref 原子，在球壳 dV 内期望找到的粒子数是：
        # dN_expected_single = (n_target / V) * dV  (若是异种原子)
        # dN_expected_single = ((n_target - 1) / V) * dV (若是同种原子，扣除自己)
        
        if elem1 == elem2:
            global_density = (n_target - 1) / vol
            # 总观测数是 n_ref，所以总的期望计数是 n_ref * global_density * dV
        else:
            global_density = n_target / vol
            
        # 3. 原始统计
        hist, _ = np.histogram(dists, bins=bin_edges)
        
        # 4. 标准化计算 g(r)
        # g(r) = 实际计数 / (参考原子数 * 全局密度 * 球壳体积)
        # 实际计数 = hist
        # 分母 = n_ref * global_density * shell_volumes
        
        norm_factor = n_ref * global_density * shell_volumes
        g_r = hist / norm_factor
        
        rdf_map[:, i] = g_r

    print("\n✅ 计算完成！")

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    time_array = np.arange(total_frames) * dt / 1000.0
    X, Y = np.meshgrid(time_array, bin_centers)
    
    # 智能设置 vmax，防止极大值掩盖细节 (取 98% 分位数)
    # 一般 g(r) 在第一配位层可能到 3-20 不等，背景是 1
    cutoff_val = np.percentile(rdf_map, 99)
    # 如果最大值太小（比如全0），就设个默认值
    if cutoff_val < 1.5: cutoff_val = 5.0
    
    plt.pcolormesh(X, Y, rdf_map, cmap='jet', shading='auto', vmin=0, vmax=cutoff_val)
    
    cbar = plt.colorbar()
    cbar.set_label('标准 RDF g(r) (趋于 1)', fontsize=12)
    
    plt.xlabel("时间 (ps)", fontsize=12)
    plt.ylabel(f"{elem1}-{elem2} 距离 (Å)", fontsize=12)
    plt.title(f"标准 g(r) 演化图: {elem1}-{elem2}", fontsize=14)
    plt.ylim(0, max_dist)
    plt.xlim(0, total_time_ps)
    
    out_file = f"Standard_RDF_{elem1}_{elem2}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 图片保存: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()
