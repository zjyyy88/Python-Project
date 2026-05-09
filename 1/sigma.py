#!/usr/bin/env python3
import argparse
import csv
import math
import re
import subprocess
from pathlib import Path

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

K_B = 1.380649e-23  # J/K
K_B_EV = 8.617333262e-5  # eV/K
E_CHARGE = 1.602176634e-19  # C
N_A = 6.02214076e23


def run_cmd(cmd: str) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stderr}")
    return proc.stdout


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_temperature(name: str) -> int | None:
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(nums[-1])


def parse_diffusivity(text: str) -> dict[int, float]:
    pattern = re.compile(
        r"(?P<name>msd[^:]*):\s*Average\s+Diffusivity\s*[:=]\s*(?P<val>[-+\d\.eE]+)",
        re.IGNORECASE,
    )
    data: dict[int, float] = {}
    for m in pattern.finditer(text):
        name = m.group("name")
        val = float(m.group("val"))
        temp = extract_temperature(name)
        if temp is None:
            continue
        data[temp] = val
    if not data:
        raise ValueError("No diffusivity entries found in output.")
    return data


def parse_concentration(text: str) -> float:
    for line in text.splitlines():
        if re.search(r"ions/cm", line, re.IGNORECASE) or re.search(r"concentration", line, re.IGNORECASE):
            vals = FLOAT_RE.findall(line)
            if vals:
                return float(vals[-1])
    raise ValueError("Cannot find ion concentration in output. Use --n to set it manually.")


def linear_fit(x: list[float], y: list[float]):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ionic conductivity and activation energy from CLI outputs."
    )
    parser.add_argument("--diffusivity-log", help="Path to diffusivity2.py output log")
    parser.add_argument("--concentration-log", help="Path to concentration.py output log")
    parser.add_argument("--diffusivity-cmd", help="Command to run diffusivity2.py")
    parser.add_argument("--concentration-cmd", help="Command to run concentration.py")
    parser.add_argument("--python", default="python", help="Python executable for default commands")
    parser.add_argument("--n", type=float, help="Ion concentration (cm^-3). Overrides parsing.")
    parser.add_argument("--z", type=float, default=1.0, help="Ion charge (default 1)")
    parser.add_argument("--fit-min", type=float, help="Min temperature (K) for fit")
    parser.add_argument("--fit-max", type=float, help="Max temperature (K) for fit")
    parser.add_argument("--no-fit", action="store_true", help="Skip activation energy fit")
    parser.add_argument("--out", default="transport_results", help="Output prefix")
    args = parser.parse_args()

    if args.diffusivity_log:
        diff_text = read_text(Path(args.diffusivity_log))
    else:
        cmd = args.diffusivity_cmd or f"{args.python} diffusivity2.py"
        diff_text = run_cmd(cmd)

    if args.n is not None:
        n_cm3 = args.n
    else:
        if args.concentration_log:
            conc_text = read_text(Path(args.concentration_log))
        else:
            cmd = args.concentration_cmd or f"{args.python} concentration.py"
            conc_text = run_cmd(cmd)
        n_cm3 = parse_concentration(conc_text)

    d_map = parse_diffusivity(diff_text)

    rows = []
    for t in sorted(d_map):
        d_a2_ps = d_map[t]
        d_cm2_s = d_a2_ps * 1e-4
        sigma = n_cm3 * (args.z * E_CHARGE) ** 2 * d_cm2_s / (K_B * t)
        rows.append(
            {
                "T_K": t,
                "D_A2_ps": d_a2_ps,
                "D_cm2_s": d_cm2_s,
                "sigma_S_cm": sigma,
                "sigma_mS_cm": sigma * 1e3,
                "ln_sigmaT": math.log(sigma * t),
                "invT_Kinv": 1.0 / t,
            }
        )

    out_prefix = Path(args.out)
    if out_prefix.suffix:
        out_prefix = out_prefix.with_suffix("")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T_K",
                "D_A2_ps",
                "D_cm2_s",
                "sigma_S_cm",
                "sigma_mS_cm",
                "ln_sigmaT",
                "invT_Kinv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    fit_path = out_prefix.with_name(out_prefix.name + "_fit.txt")
    fit_lines = [
        f"n_cm3 = {n_cm3:.6e}",
        f"z = {args.z}",
        f"points = {len(rows)}",
    ]

    if not args.no_fit:
        fit_rows = [
            r
            for r in rows
            if (args.fit_min is None or r["T_K"] >= args.fit_min)
            and (args.fit_max is None or r["T_K"] <= args.fit_max)
        ]
        if len(fit_rows) >= 2:
            x = [r["invT_Kinv"] for r in fit_rows]
            y = [r["ln_sigmaT"] for r in fit_rows]
            slope, intercept, r2 = linear_fit(x, y)
            ea_eV = -slope * K_B_EV
            ea_kj_mol = -slope * K_B * N_A / 1000.0
            fit_lines.extend(
                [
                    "", "Arrhenius fit: ln(sigma*T) vs 1/T",
                    f"fit_range_K = {fit_rows[0]['T_K']}..{fit_rows[-1]['T_K']}",
                    f"slope = {slope:.6e}",
                    f"intercept = {intercept:.6e}",
                    f"R2 = {r2:.6f}",
                    f"Ea_eV = {ea_eV:.6f}",
                    f"Ea_kJ_mol = {ea_kj_mol:.6f}",
                ]
            )
        else:
            fit_lines.append("Insufficient points for fit.")

    fit_path.write_text("\n".join(fit_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fit_path}")


if __name__ == "__main__":
    main()

