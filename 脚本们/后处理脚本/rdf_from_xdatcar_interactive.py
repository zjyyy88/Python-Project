from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.coordinates.memory import MemoryReader
from pymatgen.io.vasp import Xdatcar

"""
脚本计算逻辑（交互输入版 RDF）：
1. 运行脚本后，按提示在脚本内部输入 XDATCAR 路径和 RDF 参数。
2. 用 pymatgen 读取 XDATCAR，按 start/stop/stride 选择轨迹帧。
3. 将每帧的笛卡尔坐标和晶胞参数送入 MDAnalysis Universe。
4. 按元素名选择参考原子组 A（ref）和目标原子组 B（target）。
5. 调用 InterRDF 统计 A-B 距离分布并归一化为 g(r)。
6. 若 A 和 B 是同元素，排除自配对 i-i，避免 r≈0 的假峰。
7. 导出 r 与 g(r) 到 CSV，并输出 RDF 曲线图。
"""


def _prompt(text: str, default: str | None = None) -> str:
    if default is None:
        value = input(f"{text}: ").strip()
    else:
        value = input(f"{text} [默认 {default}]: ").strip()
        if value == "":
            value = default
    return value


def _prompt_int(text: str, default: int) -> int:
    value = _prompt(text, str(default))
    return int(value)


def _prompt_float(text: str, default: float) -> float:
    value = _prompt(text, str(default))
    return float(value)


def _prompt_optional_int(text: str) -> int | None:
    value = input(f"{text} [留空表示到最后一帧]: ").strip()
    if value == "":
        return None
    return int(value)


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
    print("=" * 60)
    print("   RDF 计算工具（交互输入版）")
    print("=" * 60)

    xdatcar_input = _prompt("请输入 XDATCAR 文件路径")
    xdatcar_path = Path(xdatcar_input)
    if not xdatcar_path.exists():
        raise FileNotFoundError(f"XDATCAR file not found: {xdatcar_path}")

    ref_elem = _prompt("请输入参考元素 ref", "Li")
    target_elem = _prompt("请输入目标元素 target", "Cl")
    r_min = _prompt_float("请输入 RDF 统计起始距离 rmin (Å)", 0.0)
    r_max = _prompt_float("请输入 RDF 统计最大距离 rmax (Å)", 8.0)
    n_bins = _prompt_int("请输入 RDF 分箱数 bins", 160)
    start = _prompt_int("请输入起始帧 start", 0)
    stop = _prompt_optional_int("请输入终止帧 stop")
    stride = _prompt_int("请输入抽帧步长 stride", 1)
    out_prefix = _prompt("请输入输出前缀 out", "rdf")

    # 主流程：读取轨迹 -> 计算 RDF -> 导出结果。
    u, available_elements = load_universe_from_xdatcar(
        xdatcar_path=xdatcar_path,
        start=start,
        stop=stop,
        stride=stride,
    )

    bins, rdf_vals, n_ref, n_target, same_species = compute_rdf(
        u,
        ref_elem=ref_elem,
        target_elem=target_elem,
        r_min=r_min,
        r_max=r_max,
        n_bins=n_bins,
    )

    out_csv = Path(f"{out_prefix}_{ref_elem}_{target_elem}.csv")
    out_png = Path(f"{out_prefix}_{ref_elem}_{target_elem}.png")

    # 第六步：输出数值表和图像，便于后续积分配位数或峰位分析。
    df = pd.DataFrame({"r_A": bins, "g_r": rdf_vals})
    df.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    plt.plot(bins, rdf_vals, lw=2)
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title(f"RDF: {ref_elem}-{target_elem}")
    plt.xlim(r_min, r_max)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print("RDF finished.")
    print(f"Frames used: {len(u.trajectory)}")
    print(f"Available elements: {available_elements}")
    print(f"Selection: {ref_elem} ({n_ref}) -> {target_elem} ({n_target})")
    if same_species:
        print("Note: same-species RDF enabled, self-pairs (i-i) were excluded.")
    print(f"CSV: {out_csv.resolve()}")
    print(f"PNG: {out_png.resolve()}")


if __name__ == "__main__":
    main()