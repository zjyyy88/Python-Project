import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Edit these defaults if you want one-click run without command-line arguments.
DEFAULT_CONFIG = {
    "structure": "D:/aaazjy/zjyyyyy/halide water adsorption/zjy-calc/Bi-dopingLYC/Li3BiCl6/QJH-CONTCAR",
    "cations": "Li,Y,Bi",
    "anions": "Cl",
    "cutoff": None,
    "cutoff_start": 2.0,
    "cutoff_end": 3.0,
    "cutoff_step": 0.1,
    #"out": str(PROJECT_ROOT / "output" / "bond_stats"),
    "out":"D:/aaazjy/zjyyyyy/halide water adsorption/zjy-calc/Bi-dopingLYC/Li3BiCl6/",
}

BOND_COLUMNS = [
    "cutoff_A",
    "cation_index",
    "cation_element",
    "anion_index",
    "anion_element",
    "coordination_number",
    "distance_A",
]

SITE_COLUMNS = [
    "cutoff_A",
    "cation_index",
    "cation_element",
    "coordination_number",
    "mean_bond_length_A",
]


def parse_element_list(raw: str) -> list[str]:
    """Parse comma-separated element symbols."""
    elems = [x.strip() for x in raw.split(",") if x.strip()]
    if not elems:
        raise ValueError("Element list is empty. Provide at least one element symbol.")
    return elems


def build_cutoff_list(
    cutoff: float | None,
    cutoff_start: float | None,
    cutoff_end: float | None,
    cutoff_step: float,
) -> list[float]:
    """Build either a single cutoff list or a cutoff scan list."""
    use_single = cutoff is not None
    use_range = cutoff_start is not None or cutoff_end is not None

    if use_single and use_range:
        raise ValueError("Use either --cutoff or (--cutoff-start/--cutoff-end), not both.")

    if use_single:
        if cutoff <= 0:
            raise ValueError("--cutoff must be > 0.")
        return [float(cutoff)]

    if cutoff_start is None or cutoff_end is None:
        raise ValueError("For cutoff scan, both --cutoff-start and --cutoff-end are required.")
    if cutoff_step <= 0:
        raise ValueError("--cutoff-step must be > 0.")
    if cutoff_start <= 0 or cutoff_end <= 0:
        raise ValueError("Cutoff values must be > 0.")
    if cutoff_end < cutoff_start:
        raise ValueError("--cutoff-end must be >= --cutoff-start.")

    n_steps = int(np.floor((cutoff_end - cutoff_start) / cutoff_step + 1e-12)) + 1
    cutoffs = [float(cutoff_start + i * cutoff_step) for i in range(n_steps)]

    # Ensure the end value is included when floating-point rounding misses it.
    if cutoffs[-1] < cutoff_end - 1e-10:
        cutoffs.append(float(cutoff_end))

    return cutoffs


def resolve_structure_path(input_path: Path) -> Path:
    """Resolve a structure path from either a file path or a directory path."""
    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        preferred = ["CONTCAR", "POSCAR"]
        for name in preferred:
            candidate = input_path / name
            if candidate.is_file():
                return candidate

        for pattern in ("*.cif", "*.vasp", "*.POSCAR", "*.CONTCAR"):
            matches = sorted(input_path.glob(pattern))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"No structure file found in directory: {input_path}. "
            "Expected CONTCAR/POSCAR or *.cif/*.vasp"
        )

    raise FileNotFoundError(f"Structure path not found: {input_path}")


def analyze_bonds(
    structure: Structure,
    cation_elements: set[str],
    anion_elements: set[str],
    cutoffs: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect all bond lengths and summarize by cation site and cutoff."""
    species_symbols = [site.specie.symbol for site in structure]

    cation_indices = [i for i, sym in enumerate(species_symbols) if sym in cation_elements]
    if not cation_indices:
        raise ValueError("No cation sites found in structure for the provided --cations.")

    anion_present = {sym for sym in species_symbols if sym in anion_elements}
    if not anion_present:
        raise ValueError("No anion sites found in structure for the provided --anions.")

    bond_rows: list[dict] = []
    site_rows: list[dict] = []

    for cutoff in cutoffs:
        for c_idx in cation_indices:
            center_site = structure[c_idx]
            neighbors = structure.get_neighbors(center_site, cutoff)

            this_site_distances = []
            this_site_bonds = []
            for nb in neighbors:
                # PeriodicNeighbor is itself a site-like object in current pymatgen versions.
                nb_symbol = nb.specie.symbol
                if nb_symbol not in anion_elements:
                    continue

                dist = float(nb.nn_distance)
                this_site_distances.append(dist)
                this_site_bonds.append(
                    {
                        "cutoff_A": cutoff,
                        "cation_index": c_idx,
                        "cation_element": center_site.specie.symbol,
                        "anion_index": int(nb.index),
                        "anion_element": nb_symbol,
                        "distance_A": dist,
                    }
                )

            this_site_cn = len(this_site_distances)
            for row in this_site_bonds:
                row["coordination_number"] = this_site_cn
                bond_rows.append(row)

            site_rows.append(
                {
                    "cutoff_A": cutoff,
                    "cation_index": c_idx,
                    "cation_element": center_site.specie.symbol,
                    "coordination_number": this_site_cn,
                    "mean_bond_length_A": float(np.mean(this_site_distances)) if this_site_distances else np.nan,
                }
            )

    bonds_df = pd.DataFrame(bond_rows, columns=BOND_COLUMNS)
    site_df = pd.DataFrame(site_rows, columns=SITE_COLUMNS)

    cutoff_rows: list[dict] = []
    for cutoff in cutoffs:
        bonds_cut = bonds_df[bonds_df["cutoff_A"] == cutoff]
        site_cut = site_df[site_df["cutoff_A"] == cutoff]

        cutoff_rows.append(
            {
                "cutoff_A": cutoff,
                "total_bonds": int(len(bonds_cut)),
                "overall_mean_bond_length_A": float(bonds_cut["distance_A"].mean()) if len(bonds_cut) else np.nan,
                "mean_coordination_number": float(site_cut["coordination_number"].mean()) if len(site_cut) else np.nan,
            }
        )

    cutoff_df = pd.DataFrame(cutoff_rows)
    return bonds_df, site_df, cutoff_df


def summarize_bond_classes(bonds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build bond-length classification tables by element pair and coordination number."""
    summary_cols = [
        "bond_count",
        "mean_bond_length_A",
        "std_bond_length_A",
        "min_bond_length_A",
        "max_bond_length_A",
    ]

    if bonds_df.empty:
        pair_df = pd.DataFrame(
            columns=["cutoff_A", "cation_element", "anion_element", "bond_type"] + summary_cols
        )
        cn_df = pd.DataFrame(columns=["cutoff_A", "cation_element", "coordination_number"] + summary_cols)
        pair_cn_df = pd.DataFrame(
            columns=["cutoff_A", "cation_element", "anion_element", "bond_type", "coordination_number"] + summary_cols
        )
        return pair_df, cn_df, pair_cn_df

    work_df = bonds_df.copy()
    work_df["bond_type"] = work_df["cation_element"] + "-" + work_df["anion_element"]

    def _group_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        grouped = (
            df.groupby(group_cols, dropna=False)["distance_A"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        return grouped.rename(
            columns={
                "count": "bond_count",
                "mean": "mean_bond_length_A",
                "std": "std_bond_length_A",
                "min": "min_bond_length_A",
                "max": "max_bond_length_A",
            }
        )

    pair_df = _group_stats(work_df, ["cutoff_A", "cation_element", "anion_element", "bond_type"])
    cn_df = _group_stats(work_df, ["cutoff_A", "cation_element", "coordination_number"])
    pair_cn_df = _group_stats(
        work_df,
        ["cutoff_A", "cation_element", "anion_element", "bond_type", "coordination_number"],
    )

    return pair_df, cn_df, pair_cn_df


def resolve_output_prefix(out_arg: str) -> Path:
    """Return a usable output prefix; supports both directory path and file prefix."""
    out_raw = out_arg.strip()
    out_path = Path(out_raw)
    out_is_dir = out_raw.endswith(("/", "\\")) or out_path.is_dir()

    if out_is_dir:
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / "bond_stats"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count anion coordination around cation centers within adjustable cutoff(s), "
            "and export all bond lengths plus average bond lengths."
        )
    )
    parser.add_argument(
        "--structure",
        default=DEFAULT_CONFIG["structure"],
        help="Input structure file (default from DEFAULT_CONFIG)",
    )
    parser.add_argument(
        "--cations",
        default=DEFAULT_CONFIG["cations"],
        help="Comma-separated cation elements (default from DEFAULT_CONFIG)",
    )
    parser.add_argument(
        "--anions",
        default=DEFAULT_CONFIG["anions"],
        help="Comma-separated anion elements (default from DEFAULT_CONFIG)",
    )

    parser.add_argument("--cutoff", type=float, default=None, help="Single cutoff radius in Angstrom")
    parser.add_argument("--cutoff-start", type=float, default=None, help="Cutoff scan start in Angstrom")
    parser.add_argument("--cutoff-end", type=float, default=None, help="Cutoff scan end in Angstrom")
    parser.add_argument(
        "--cutoff-step",
        type=float,
        default=DEFAULT_CONFIG["cutoff_step"],
        help="Cutoff scan step in Angstrom (default from DEFAULT_CONFIG)",
    )

    parser.add_argument("--out", default=DEFAULT_CONFIG["out"], help="Output prefix (default from DEFAULT_CONFIG)")
    args = parser.parse_args()

    # If user runs with no cutoff args, fall back to built-in scan defaults.
    if args.cutoff is None and args.cutoff_start is None and args.cutoff_end is None:
        args.cutoff = DEFAULT_CONFIG["cutoff"]
        args.cutoff_start = DEFAULT_CONFIG["cutoff_start"]
        args.cutoff_end = DEFAULT_CONFIG["cutoff_end"]

    structure_path = resolve_structure_path(Path(args.structure))

    structure = Structure.from_file(str(structure_path))
    cation_elements = set(parse_element_list(args.cations))
    anion_elements = set(parse_element_list(args.anions))
    cutoffs = build_cutoff_list(args.cutoff, args.cutoff_start, args.cutoff_end, args.cutoff_step)

    bonds_df, site_df, cutoff_df = analyze_bonds(
        structure=structure,
        cation_elements=cation_elements,
        anion_elements=anion_elements,
        cutoffs=cutoffs,
    )
    pair_df, cn_df, pair_cn_df = summarize_bond_classes(bonds_df)

    out_prefix = resolve_output_prefix(args.out)
    bonds_csv = Path(f"{out_prefix}_all_bonds.csv")
    site_csv = Path(f"{out_prefix}_site_summary.csv")
    cutoff_csv = Path(f"{out_prefix}_cutoff_summary.csv")
    pair_csv = Path(f"{out_prefix}_bond_type_summary.csv")
    cn_csv = Path(f"{out_prefix}_coordination_summary.csv")
    pair_cn_csv = Path(f"{out_prefix}_bond_type_coordination_summary.csv")

    bonds_df.to_csv(bonds_csv, index=False, encoding="utf-8")
    site_df.to_csv(site_csv, index=False, encoding="utf-8")
    cutoff_df.to_csv(cutoff_csv, index=False, encoding="utf-8")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8")
    cn_df.to_csv(cn_csv, index=False, encoding="utf-8")
    pair_cn_df.to_csv(pair_cn_csv, index=False, encoding="utf-8")

    print("Bond statistics finished.")
    print(f"Input structure: {structure_path.resolve()}")
    print(f"Cations: {sorted(cation_elements)}")
    print(f"Anions: {sorted(anion_elements)}")
    print(f"Number of cation sites: {site_df['cation_index'].nunique()}")
    print(f"Cutoffs (A): {cutoffs}")
    print(f"All bonds CSV: {bonds_csv.resolve()}")
    print(f"Site summary CSV: {site_csv.resolve()}")
    print(f"Cutoff summary CSV: {cutoff_csv.resolve()}")
    print(f"Bond type summary CSV: {pair_csv.resolve()}")
    print(f"Coordination summary CSV: {cn_csv.resolve()}")
    print(f"Bond type + coordination summary CSV: {pair_cn_csv.resolve()}")

    for _, row in cutoff_df.iterrows():
        mean_len = row["overall_mean_bond_length_A"]
        mean_cn = row["mean_coordination_number"]
        print(
            f"cutoff={row['cutoff_A']:.3f} A | "
            f"total_bonds={int(row['total_bonds'])} | "
            f"overall_mean_bond_length={mean_len:.4f} A | "
            f"mean_CN={mean_cn:.3f}"
        )


if __name__ == "__main__":
    main()
