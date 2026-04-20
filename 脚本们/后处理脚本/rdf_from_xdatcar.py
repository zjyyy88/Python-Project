import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.coordinates.memory import MemoryReader
from pymatgen.io.vasp import Xdatcar

"""
脚本计算逻辑（RDF）：
1. 用 pymatgen 读取 XDATCAR，按 start/stop/stride 选择轨迹帧。
2. 将每帧的笛卡尔坐标和晶胞参数 [a,b,c,alpha,beta,gamma] 送入 MDAnalysis Universe。
3. 按元素名选择参考原子组 A（ref）和目标原子组 B（target）。
4. 调用 InterRDF 统计 A-B 距离分布并归一化为 g(r)。
5. 若 A 和 B 是同元素，排除自配对 i-i，避免 r≈0 的假峰。
6. 导出 r 与 g(r) 到 CSV，并输出 RDF 曲线图。
"""

"""
先按 start/stop/stride 选出一组帧。
InterRDF.run() 会遍历这些帧累计统计并归一化。
最终输出的 g(r) 相当于所有选定帧的平均 RDF，适合后续积分配位数或峰位分析。

默认参数下（start=0, stop=None, stride=1）就是全轨迹平均。
如果你只想看单帧，把参数设成例如 start=200, stop=201（只取第 200 帧）。
目前脚本输出的是一条平均曲线，不是逐帧时间分辨的 RDF。
"""

def load_universe_from_xdatcar(xdatcar_path: Path, start: int, stop: int | None, stride: int):
    # 第一步：读取 XDATCAR 并按用户给定的帧区间抽样。
    xd = Xdatcar(str(xdatcar_path))
    structures = xd.structures[start:stop:stride]
    if not structures:
        raise ValueError("No frames selected. Check start/stop/stride.")

    # 第二步：组织轨迹数组。
    # coords 形状：(n_frames, n_atoms, 3)
    n_atoms = len(structures[0])
    coords = np.asarray([s.cart_coords for s in structures], dtype=np.float32)
    # dimensions 使用 MDAnalysis 三斜晶胞约定：[a,b,c,alpha,beta,gamma]
    dims = np.asarray(
        [
            [s.lattice.a, s.lattice.b, s.lattice.c, s.lattice.alpha, s.lattice.beta, s.lattice.gamma]
            for s in structures
        ],
        dtype=np.float32,
    )
    names = [site.specie.symbol for site in structures[0]]

    # 第三步：在内存中构建 Universe，避免中间格式转换文件。
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", names)
    u.load_new(coords, format=MemoryReader, order="fac", dimensions=dims)

    return u, sorted(set(names))


def compute_rdf(u, ref_elem: str, target_elem: str, r_min: float, r_max: float, n_bins: int):
    # 第四步：按元素名选择 A/B 两组原子。
    ag1 = u.select_atoms(f"name {ref_elem}")
    ag2 = u.select_atoms(f"name {target_elem}")

    if ag1.n_atoms == 0:
        raise ValueError(f"Reference group empty: {ref_elem}")
    if ag2.n_atoms == 0:
        raise ValueError(f"Target group empty: {target_elem}")

    # 同元素 RDF（A-A）时排除 i-i 自配对，避免 r=0 的非物理尖峰。
    same_species = ref_elem == target_elem
    exclusion_block = (1, 1) if same_species else None

    # 第五步：统计距离直方图并归一化为 g(r)。
    # norm="rdf" 表示按理想气体基准进行标准 RDF 归一化。
    rdf = InterRDF(
        ag1,
        ag2,
        nbins=n_bins,
        range=(r_min, r_max),
        exclusion_block=exclusion_block,
        norm="rdf",
    )
    rdf.run()

    return rdf.results.bins, rdf.results.rdf, ag1.n_atoms, ag2.n_atoms, same_species


def main():
    parser = argparse.ArgumentParser(description="Compute RDF from VASP XDATCAR using MDAnalysis.")
    parser.add_argument("--xdatcar", required=True, help="Path to XDATCAR file")
    parser.add_argument("--ref", default="Li", help="Reference element (default: Li)")
    parser.add_argument("--target", default="Cl", help="Target element (default: Cl)")
    parser.add_argument("--rmin", type=float, default=0.0, help="RDF range minimum in Angstrom")
    parser.add_argument("--rmax", type=float, default=8.0, help="RDF range maximum in Angstrom")
    parser.add_argument("--bins", type=int, default=160, help="Number of RDF bins")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--out", default="rdf", help="Output prefix")
    args = parser.parse_args()

    xdatcar_path = Path(args.xdatcar)
    if not xdatcar_path.exists():
        raise FileNotFoundError(f"XDATCAR file not found: {xdatcar_path}")

    # 主流程：读取轨迹 -> 计算 RDF -> 导出结果。
    u, available_elements = load_universe_from_xdatcar(
        xdatcar_path=xdatcar_path,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )

    bins, rdf_vals, n_ref, n_target, same_species = compute_rdf(
        u,
        ref_elem=args.ref,
        target_elem=args.target,
        r_min=args.rmin,
        r_max=args.rmax,
        n_bins=args.bins,
    )

    out_csv = Path(f"{args.out}_{args.ref}_{args.target}.csv")
    out_png = Path(f"{args.out}_{args.ref}_{args.target}.png")

    # 第六步：输出数值表和图像，便于后续积分配位数或峰位分析。
    df = pd.DataFrame({"r_A": bins, "g_r": rdf_vals})
    df.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    plt.plot(bins, rdf_vals, lw=2)
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title(f"RDF: {args.ref}-{args.target}")
    plt.xlim(args.rmin, args.rmax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print("RDF finished.")
    print(f"Frames used: {len(u.trajectory)}")
    print(f"Available elements: {available_elements}")
    print(f"Selection: {args.ref} ({n_ref}) -> {args.target} ({n_target})")
    if same_species:
        print("Note: same-species RDF enabled, self-pairs (i-i) were excluded.")
    print(f"CSV: {out_csv.resolve()}")
    print(f"PNG: {out_png.resolve()}")


if __name__ == "__main__":
    main()