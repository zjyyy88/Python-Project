#!/usr/bin/env python3
"""
从 XDATCAR 读取轨迹并计算 MSD（支持 Direct/Cartesian，自动 unwrap）。

用法示例：
  python MSD-fromXDATCAR.py --xdatcar XDATCAR --species Li --timestep_fs 20

注意：`--timestep_fs` 单位为 fs（例如 POTIM=2, NBLOCK=10 -> 每帧 20 fs）。
"""
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_xdatcar(path):
    path = Path(path)
    with path.open("r", errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip() != ""]
    if len(lines) < 10:
        raise ValueError("XDATCAR 内容过短，无法解析")
    scale = float(lines[1].split()[0])
    lattice = np.array([list(map(float, lines[2 + i].split())) for i in range(3)]) * scale
    tok5 = lines[5].split()
    if all(not t.replace('.', '', 1).isdigit() for t in tok5):
        elements = tok5
        counts = list(map(int, lines[6].split()))
        coord_start = 8
    else:
        elements = None
        counts = list(map(int, tok5))
        coord_start = 7
    coords_type = lines[coord_start - 1].strip()
    raw = []
    for ln in lines[coord_start:]:
        l = ln.strip()
        if l.lower().startswith("direct") or l.lower().startswith("cart") or "configuration" in l.lower():
            continue
        raw.append(l)
    total_atoms = sum(counts)
    if total_atoms == 0:
        raise ValueError("解析到的原子数为零，请检查 XDATCAR header")
    if len(raw) % total_atoms != 0:
        raise ValueError(f"Parsed coordinate lines not divisible by atom count ({len(raw)} vs {total_atoms})")
    nframes = len(raw) // total_atoms
    frames = np.zeros((nframes, total_atoms, 3), dtype=float)
    for fi in range(nframes):
        for ai in range(total_atoms):
            parts = raw[fi * total_atoms + ai].split()[:3]
            frames[fi, ai, :] = list(map(float, parts))
    return lattice, elements, counts, coords_type, frames


def make_species_list(elements, counts):
    if elements is None:
        species = []
        for i, c in enumerate(counts):
            species += [f"X{i}"] * c
        return species
    species = []
    for el, c in zip(elements, counts):
        species += [el] * c
    return species


def unwrap_frac(frac_frames):
    nframes, natoms, _ = frac_frames.shape
    unwrapped = np.zeros_like(frac_frames)
    unwrapped[0] = frac_frames[0]
    for t in range(1, nframes):
        delta = frac_frames[t] - frac_frames[t - 1]
        delta = delta - np.round(delta)
        unwrapped[t] = unwrapped[t - 1] + delta
    return unwrapped


def compute_msd(cart_frames, indices):
    pos0 = cart_frames[0, indices, :]
    disp = cart_frames[:, indices, :] - pos0[np.newaxis, :, :]
    sq = np.sum(disp ** 2, axis=2)
    msd = np.mean(sq, axis=1)
    msd_x = np.mean(disp[:, :, 0] ** 2, axis=1)
    msd_y = np.mean(disp[:, :, 1] ** 2, axis=1)
    msd_z = np.mean(disp[:, :, 2] ** 2, axis=1)
    return msd, msd_x, msd_y, msd_z


def main():
    parser = argparse.ArgumentParser(description="Compute MSD from XDATCAR")
    parser.add_argument("--xdatcar", default="XDATCAR", help="XDATCAR path")
    parser.add_argument("--species", default="Li", help="Target species label (e.g., Li)")
    parser.add_argument("--timestep_fs", type=float, default=10, help="Timestep per frame in fs (for time axis)")
    parser.add_argument("--fit_start", type=int, default=1, help="Fit start index (1-based index)")
    parser.add_argument("--fit_end", type=int, default=5000, help="Fit end index (inclusive, 1-based)")
    parser.add_argument("--out_prefix", default="XDATCAR_MSD", help="Output filename prefix")
    args = parser.parse_args()

    lattice, elements, counts, coords_type, frames = parse_xdatcar(args.xdatcar)
    species = make_species_list(elements, counts)
    total_atoms = sum(counts)
    nframes = frames.shape[0]
    if coords_type.lower().startswith("direct"):
        frac = frames.copy()
    else:
        inv_lat = np.linalg.inv(lattice)
        frac = np.einsum("fab,bc->fac", frames, inv_lat)
    unwrapped_frac = unwrap_frac(frac)
    cart = np.einsum("fab,bc->fac", unwrapped_frac, lattice)
    indices = [i for i, s in enumerate(species) if s == args.species]
    if len(indices) == 0:
        raise ValueError(f"No atoms found for species '{args.species}'")
    msd, msd_x, msd_y, msd_z = compute_msd(cart, indices)
    time_ps = np.arange(nframes) * (args.timestep_fs / 1000.0)
    start = max(0, args.fit_start - 1)
    end = min(nframes, args.fit_end)
    coef = np.polyfit(time_ps[start:end], msd[start:end], 1)
    slope = coef[0]
    D_cm2_s = slope / 6.0 * 1e-4
    df = pd.DataFrame({
        "time_ps": time_ps,
        f"{args.species}_MSD_A2": msd,
        f"{args.species}_MSD_x_A2": msd_x,
        f"{args.species}_MSD_y_A2": msd_y,
        f"{args.species}_MSD_z_A2": msd_z,
    })
    csvp = f"{args.out_prefix}_{args.species}_MSD.csv"
    df.to_csv(csvp, index=False)
    print(f"Saved MSD -> {csvp}")
    print(f"Slope (Å^2/ps): {slope:.6e}")
    print(f"Estimated D (cm^2/s): {D_cm2_s:.6e}")
    plt.figure(figsize=(6, 4))
    plt.plot(time_ps, msd, label="MSD")
    plt.plot(time_ps[start:end], np.poly1d(coef)(time_ps[start:end]), "--", label="fit")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å²)")
    plt.legend()
    plt.tight_layout()
    pngp = f"{args.out_prefix}_{args.species}_MSD.png"
    plt.savefig(pngp, dpi=300)
    print(f"Plot saved -> {pngp}")


if __name__ == "__main__":
    main()
