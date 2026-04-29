"""基于 DiffusionAnalyzer 逻辑重新绘制 MSD 图。

优先使用与 Liconductivity.py 一致的流程：
1) 读取 XDATCAR
2) DiffusionAnalyzer.from_structures(...)
3) export_msdt 导出 msd 数据
4) 读取 msd 数据并绘图

也支持直接传入已有 msd 数据文件重绘（--msd-file）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.core.trajectory import Trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 DiffusionAnalyzer 逻辑重绘 MSD 图")
    parser.add_argument("--xdatcar", type=Path, default=Path("XDATCAR"), help="XDATCAR 路径")
    parser.add_argument("--specie", default="Li", help="离子种类，例如 Li")
    parser.add_argument("--temperature", type=float, default=1900.0, help="温度 (K)")
    parser.add_argument("--time-step", type=int, default=2, help="POTIM/fs")
    parser.add_argument("--step-skip", type=int, default=1, help="轨迹抽样步长")
    parser.add_argument(
        "--msd-file",
        type=Path,
        default=Path("msd.dat"),
        help="MSD 数据文件路径；若不存在则尝试由 XDATCAR 生成",
    )
    parser.add_argument("--outfig", type=Path, default=Path("msd_replot.png"), help="输出图文件")
    parser.add_argument("--show", action="store_true", help="是否弹窗显示")
    return parser.parse_args()


def ensure_msd_file(args: argparse.Namespace) -> Path:
    msd_file = args.msd_file
    if msd_file.exists():
        return msd_file

    if not args.xdatcar.exists():
        raise FileNotFoundError(f"未找到 MSD 文件且 XDATCAR 不存在: {args.xdatcar}")

    traj = Trajectory.from_file(str(args.xdatcar))
    diff = DiffusionAnalyzer.from_structures(
        traj,
        args.specie,
        args.temperature,
        args.time_step,
        args.step_skip,
    )
    diff.export_msdt(str(msd_file))
    return msd_file


def load_msd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # 兼容两种常见格式：纯数值两列，或首行包含文本表头。
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.genfromtxt(path, skip_header=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"MSD 文件列数不足: {path}")

    time = data[:, 0]
    msd = data[:, 1]
    return time, msd


def main() -> None:
    args = parse_args()
    msd_file = ensure_msd_file(args)
    time, msd = load_msd(msd_file)

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    ax.plot(time, msd, color="#2E86AB", linewidth=2.0, label=f"{args.specie} MSD")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (A$^2$)")
    ax.set_title(f"{args.specie} MSD at {args.temperature:g} K")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(args.outfig, dpi=300, bbox_inches="tight")

    if args.show:
        plt.show()
    plt.close(fig)

    print(f"MSD 数据来源: {msd_file}")
    print(f"图像已保存: {args.outfig}")
    print(f"数据点数: {len(time)}")


if __name__ == "__main__":
    main()
