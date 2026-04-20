import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.io.vasp import Xdatcar

"""
RDF/CN check logic (rough sanity evaluation):

1) Read frame-0 from XDATCAR and estimate target number density:
        rho_B = N_B / V
    For same-species A-A pair, use effective N_B = N_A - 1 to exclude self-neighbor.

2) Load RDF table (r, g(r)) and clean/sort numeric values.

3) Build cumulative coordination number curve:
        CN(r) = 4 * pi * rho_B * integral_0^r [ g(r') * (r')^2 dr' ]
    Numerically, this script uses cumulative trapezoidal integration.

4) Choose first-shell cutoff r_cut:
    - user-provided --rcut, or
    - first local minimum after the first major peak of g(r).

5) Report CN(r_cut) and compare with random baseline:
        CN_random(r_cut) = (4/3) * pi * r_cut^3 * rho_B

6) Quality hint: check whether RDF tail approaches ~1.
"""


def load_first_frame_info(xdatcar_path: Path, ref_elem: str, target_elem: str):
     # Use only frame-0 composition and volume for a quick density estimate.
    xd = Xdatcar(str(xdatcar_path))
    if not xd.structures:
        raise ValueError("XDATCAR contains no structures.")

    st = xd.structures[0]
    volume = float(st.volume)
    species = [site.specie.symbol for site in st]

    n_ref = sum(1 for s in species if s == ref_elem)
    n_target = sum(1 for s in species if s == target_elem)

    if n_ref == 0:
        raise ValueError(f"Reference element not found in frame 0: {ref_elem}")
    if n_target == 0:
        raise ValueError(f"Target element not found in frame 0: {target_elem}")

    same_species = ref_elem == target_elem
    if same_species:
        if n_target <= 1:
            raise ValueError("Need at least 2 atoms for same-species RDF/CN evaluation.")
        # Exclude self-pair contribution for A-A case.
        effective_n_target = n_target - 1
    else:
        effective_n_target = n_target

    rho_b = effective_n_target / volume

    return {
        "n_atoms": len(st),
        "n_ref": n_ref,
        "n_target": n_target,
        "effective_n_target": effective_n_target,
        "volume_A3": volume,
        "rho_b_A3": rho_b,
        "same_species": same_species,
    }


def _pick_rdf_columns(df: pd.DataFrame):
    # Prefer explicit headers from our RDF exporter.
    expected = {"r_A", "g_r"}
    if expected.issubset(df.columns):
        return "r_A", "g_r"

    numeric_cols = []
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        raise ValueError(
            "Cannot identify RDF columns. Need at least two numeric columns or explicit columns r_A and g_r."
        )

    return numeric_cols[0], numeric_cols[1]


def load_rdf_csv(rdf_csv: Path):
    df = pd.read_csv(rdf_csv)
    r_col, g_col = _pick_rdf_columns(df)

    r = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(df[g_col], errors="coerce").to_numpy(dtype=float)

    # Keep finite, physical distances only.
    mask = np.isfinite(r) & np.isfinite(g) & (r >= 0.0)
    r = r[mask]
    g = g[mask]

    if r.size < 3:
        raise ValueError("RDF data points are too few after cleaning.")

    # Ensure monotonic r before numerical integration/interpolation.
    order = np.argsort(r)
    r = r[order]
    g = g[order]

    return r, g


def cumulative_trapz(y: np.ndarray, x: np.ndarray):
    # Cumulative trapezoid integration with y=f(x), returns integral from x[0] to each x[i].
    if x.size < 2:
        return np.zeros_like(x)
    area = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(area)))


def find_first_shell_cutoff(r: np.ndarray, g: np.ndarray, peak_search_start: float):
    # Heuristic: first-shell boundary = first local minimum after first major peak.
    candidates = np.where(r >= peak_search_start)[0]
    if candidates.size == 0:
        return None, None

    peak_start = int(candidates[0])
    peak_idx = peak_start + int(np.argmax(g[peak_start:]))

    if peak_idx >= r.size - 2:
        return peak_idx, None

    dg = np.diff(g)
    min_idx = None

    # Find sign change in derivative: negative -> non-negative.
    for i in range(peak_idx + 1, r.size - 1):
        if dg[i - 1] < 0 and dg[i] >= 0:
            min_idx = i
            break

    if min_idx is None:
        min_idx = peak_idx + 1 + int(np.argmin(g[peak_idx + 1 :]))

    return peak_idx, min_idx


def main():
    parser = argparse.ArgumentParser(
        description="RDF sanity check by coordination-number integration using XDATCAR frame-0 density."
    )
    parser.add_argument("--xdatcar", required=True, help="Path to XDATCAR file")
    parser.add_argument("--rdf", required=True, help="Path to RDF csv (must contain r and g(r) columns)")
    parser.add_argument("--ref", required=True, help="Reference element, e.g., Li")
    parser.add_argument("--target", required=True, help="Target element, e.g., Cl")
    parser.add_argument("--rcut", type=float, default=None, help="Cutoff radius for CN integration (A)")
    parser.add_argument(
        "--peak-start",
        type=float,
        default=1.0,
        help="Start radius to search first major RDF peak (A), default 1.0",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.2,
        help="Last fraction of RDF points used to check g(r) tail ~ 1, default 0.2",
    )
    parser.add_argument("--out", default="rdf_check", help="Output prefix")
    args = parser.parse_args()

    xdatcar_path = Path(args.xdatcar)
    rdf_path = Path(args.rdf)

    if not xdatcar_path.exists():
        raise FileNotFoundError(f"XDATCAR not found: {xdatcar_path}")
    if not rdf_path.exists():
        raise FileNotFoundError(f"RDF csv not found: {rdf_path}")

    # Step 1: density estimate from frame-0 only.
    info = load_first_frame_info(xdatcar_path, args.ref, args.target)
    # Step 2: load RDF curve to be checked.
    r, g = load_rdf_csv(rdf_path)

    # Step 3: CN(r) = 4*pi*rho_B*integral(g(r)*r^2 dr)
    integrand = g * (r**2)
    cn = 4.0 * np.pi * info["rho_b_A3"] * cumulative_trapz(integrand, r)

    # Step 4: determine first-shell cutoff radius.
    peak_idx, auto_min_idx = find_first_shell_cutoff(r, g, peak_search_start=args.peak_start)

    if args.rcut is not None:
        r_cut = float(args.rcut)
    elif auto_min_idx is not None:
        r_cut = float(r[auto_min_idx])
    else:
        r_cut = float(r[-1])

    # Step 5: report CN at cutoff and compare to random baseline in same sphere volume.
    cn_at_rcut = float(np.interp(r_cut, r, cn))
    cn_random_at_rcut = float((4.0 / 3.0) * np.pi * (r_cut**3) * info["rho_b_A3"])

    # Step 6: quick normalization sanity check; ideal homogeneous tail is near g(r)=1.
    tail_n = max(3, int(np.ceil(r.size * args.tail_fraction)))
    tail_vals = g[-tail_n:]
    tail_mean = float(np.mean(tail_vals))
    tail_std = float(np.std(tail_vals))

    out_curve = Path(f"{args.out}_{args.ref}_{args.target}_cn_curve.csv")
    out_summary = Path(f"{args.out}_{args.ref}_{args.target}_summary.txt")

    pd.DataFrame({"r_A": r, "g_r": g, "cn_int": cn}).to_csv(out_curve, index=False)

    lines = [
        "RDF coordination-number check (rough estimate from XDATCAR frame 0)",
        f"xdatcar: {xdatcar_path}",
        f"rdf_csv: {rdf_path}",
        f"pair: {args.ref}-{args.target}",
        "",
        f"frame0_n_atoms = {info['n_atoms']}",
        f"frame0_volume_A3 = {info['volume_A3']:.6f}",
        f"n_ref = {info['n_ref']}",
        f"n_target = {info['n_target']}",
        f"effective_n_target_for_density = {info['effective_n_target']}",
        f"rho_B_A^-3 = {info['rho_b_A3']:.8f}",
        f"same_species = {info['same_species']}",
        "",
        f"first_peak_r_A = {float(r[peak_idx]):.4f}" if peak_idx is not None else "first_peak_r_A = N/A",
        (
            f"auto_first_min_r_A = {float(r[auto_min_idx]):.4f}" if auto_min_idx is not None else "auto_first_min_r_A = N/A"
        ),
        f"used_rcut_A = {r_cut:.4f}",
        f"CN_integrated_at_rcut = {cn_at_rcut:.6f}",
        f"CN_random_at_rcut = {cn_random_at_rcut:.6f}",
        f"CN_ratio_to_random = {cn_at_rcut / cn_random_at_rcut:.6f}" if cn_random_at_rcut > 0 else "CN_ratio_to_random = N/A",
        "",
        f"g_tail_mean(last_{tail_n}_pts) = {tail_mean:.6f}",
        f"g_tail_std(last_{tail_n}_pts) = {tail_std:.6f}",
    ]

    # Simple sanity flags for quick screening.
    flags = []
    if abs(tail_mean - 1.0) > 0.25:
        flags.append("Tail mean of g(r) is far from 1.0; check equilibration/rcut/normalization.")
    if cn_at_rcut <= 0:
        flags.append("Integrated CN is non-positive; check pair selection and RDF data.")
    if auto_min_idx is None and args.rcut is None:
        flags.append("Could not locate first-shell minimum automatically; set --rcut manually.")

    if flags:
        lines.append("")
        lines.append("flags:")
        lines.extend([f"- {f}" for f in flags])

    out_summary.write_text("\n".join(lines), encoding="utf-8")

    print("RDF CN check finished.")
    print(f"Curve CSV: {out_curve.resolve()}")
    print(f"Summary : {out_summary.resolve()}")
    print(f"rho_B (A^-3): {info['rho_b_A3']:.8f}")
    print(f"used rcut (A): {r_cut:.4f}")
    print(f"CN(rcut): {cn_at_rcut:.6f}")
    print(f"g_tail mean/std: {tail_mean:.4f}/{tail_std:.4f}")


if __name__ == "__main__":
    main()
