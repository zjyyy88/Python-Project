"""计算基于多个时间原点的 MSD（time-origin averaged MSD），并可选择线性拟合区间。

特点：
- 支持 vasprun.xml 或 XDATCAR 输入
- 使用最小镜像约定处理分数坐标
- 默认限制最大滞后帧以控制计算量（可通过 --max-lag 设置）
- 输出：数据文件（tau(ps), msd(A^2)）、PNG 图像、可选线性拟合结果（并返回 D）

用法示例：
python MSD_time_origin_average.py --input path/to/vasprun.xml --specie Li --time-step 0.001 --max-lag 500 --fit-range 100 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp import Vasprun


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-origin averaged MSD")
    p.add_argument("--input", type=Path, required=True, help="vasprun.xml 或 XDATCAR 文件路径")
    p.add_argument("--specie", default="Li", help="要统计的物种，比如 Li")
    p.add_argument("--time-step", type=float, default=1.0, help="每帧对应的时间（ps），若 vasprun 可用真实值则输入真实值")
    p.add_argument("--max-lag", type=int, default=None, help="最大滞后帧数（默认取 min(500, N//4)）")
    p.add_argument("--fit-range", nargs=2, type=float, default=None, metavar=("TMIN","TMAX"), help="拟合线性的时间区间，单位 ps")
    p.add_argument("--out-prefix", default=None, help="输出文件名前缀（默认用物种名）")
    p.add_argument("--show", action="store_true", help="显示图像窗口")
    return p.parse_args()


def load_traj(path: Path):
    if path.suffix.lower() == ".xml":
        vr = Vasprun(str(path), parse_dos=False, parse_eigen=False)
        return list(vr.get_trajectory())
    return list(Trajectory.from_file(str(path)))


def select_indices(structure, specie: str) -> list[int]:
    return [i for i, s in enumerate(structure) if str(s.specie).startswith(specie)]


def minimum_image_frac(delta_frac: np.ndarray) -> np.ndarray:
    return delta_frac - np.round(delta_frac)


def compute_time_origin_msd(traj, indices: list[int], max_lag: int | None = None):
    N = len(traj)
    if max_lag is None:
        max_lag = min(500, N // 4)
    else:
        max_lag = min(max_lag, N - 1)

    # build fractional coordinate array and lattice matrices
    frac = np.array([[frame[i].frac_coords for i in indices] for frame in traj])  # shape (N, nions, 3)
    lattices = np.array([frame.lattice.matrix for frame in traj])  # shape (N, 3, 3)

    taus = np.arange(0, max_lag + 1)
    msd = np.zeros_like(taus, dtype=float)

    nions = len(indices)

    for k, tau in enumerate(taus):
        if tau == 0:
            msd[k] = 0.0
            continue
        # origins: 0 .. N-1-tau
        n_origins = N - tau
        # delta_frac shape (n_origins, nions, 3)
        delta_frac = frac[tau :, :, :] - frac[: n_origins, :, :]
        # apply minimum image per origin
        delta_frac = minimum_image_frac(delta_frac)
        # convert to cartesian using lattice at origin frames
        # broadcast: (n_origins, 3,3) @ (n_origins, nions,3).T --> (n_origins, nions,3)
        cart = np.einsum("ijk,ikm->ijm", lattices[:n_origins], delta_frac)
        # squared norms per origin and ion
        sq = np.sum(cart * cart, axis=2)  # (n_origins, nions)
        # average over ions then origins
        msd[k] = float(np.mean(sq))

    return taus, msd


def fit_linear(t_ps: np.ndarray, msd: np.ndarray, tmin: float, tmax: float):
    mask = (t_ps >= tmin) & (t_ps <= tmax)
    if mask.sum() < 2:
        return None
    coeff = np.polyfit(t_ps[mask], msd[mask], 1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    # convert slope (A^2/ps) to D in cm^2/s: D = slope/(6) * 1e-4
    D_cm2_s = slope / 6.0 * 1e-4
    return slope, intercept, D_cm2_s


def main():
    args = parse_args()
    traj = load_traj(args.input)
    if len(traj) < 2:
        raise SystemExit("轨迹帧不足")

    indices = select_indices(traj[0], args.specie)
    if not indices:
        raise SystemExit(f"未找到物种 {args.specie}")

    N = len(traj)
    max_lag_default = min(500, N // 4)
    taus, msd = compute_time_origin_msd(traj, indices, max_lag=args.max_lag or max_lag_default)

    time_ps = taus * args.time_step
    prefix = args.out_prefix or args.specie
    out_data = Path(f"{prefix}_msd_time_origin.dat")
    np.savetxt(out_data, np.column_stack([time_ps, msd]), header="time_ps\tmsd_A2")

    # plot
    plt.figure(figsize=(7.2, 4.8), dpi=300)
    plt.plot(time_ps, msd, '-', linewidth=1, label=f"{args.specie} MSD (time-origin averaged)")

    fit_res = None
    if args.fit_range is not None:
        slope, intercept, D = fit_linear(time_ps, msd, args.fit_range[0], args.fit_range[1])
        fit_res = (slope, intercept, D)
        if slope is not None:
            plt.plot(time_ps, slope * time_ps + intercept, '--', color='C3', label=f'Linear fit {args.fit_range[0]}-{args.fit_range[1]} ps')

    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (A$^2$)')
    plt.title(f"{args.specie} MSD ({len(traj)} frames, {len(indices)} atoms)")
    plt.legend()
    plt.grid(alpha=0.2)
    out_fig = Path(f"{prefix}_msd_time_origin.png")
    plt.tight_layout()
    plt.savefig(out_fig)
    if args.show:
        plt.show()
    plt.close()

    print(f"Saved MSD data: {out_data}")
    print(f"Saved MSD figure: {out_fig}")
    if fit_res is not None:
        slope, intercept, D = fit_res
        print(f"Linear slope (A^2/ps): {slope:.6e}, intercept: {intercept:.6e}")
        print(f"Estimated D: {D:.6e} cm^2/s (from slope/6)")


if __name__ == '__main__':
    main()
