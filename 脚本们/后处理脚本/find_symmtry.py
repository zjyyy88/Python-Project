#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch symmetry analysis for VASP files and CIF export.

This script combines ASE and pymatgen:
1) ASE reads VASP structures (*.vasp)
2) pymatgen analyzes symmetry and standardizes structures
3) CIF files with symmetry information are written in batch

Example:
    python symmetry_export_cif.py \
        --pattern "E:\\固态组\\LiLa2O3\\LixMgxLa32-xO48\\掺杂base结构\\*.vasp" \
        --out-dir "E:\\固态组\\LiLa2O3\\LixMgxLa32-xO48\\掺杂base结构\\symmetry_cif"
"""

from __future__ import annotations

import argparse
import csv
import re
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

DEFAULT_PATTERN = r"E:固态组\LiLa2O3\La2O3-Lithiation\LiLa2O3-wzSEND.vasp"
IA3_SYMBOL = "Ia-3"
IA3_NUMBER = 206
DEFAULT_IA3_SCAN = "0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find symmetry for VASP structures and export symmetry-aware CIF files."
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=None,
        help=(
            "Glob pattern for input VASP files. Can be passed multiple times. "
            "Default is your target path pattern."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="symmetry_cif_outputs",
        help="Directory to save exported CIF files and the CSV summary.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.01,
        help="Symmetry finding tolerance in angstrom.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=5.0,
        help="Angle tolerance in degrees for symmetry finding.",
    )
    parser.add_argument(
        "--structure-mode",
        choices=["refined", "conventional", "primitive", "input"],
        default="refined",
        help=(
            "Structure form to export to CIF. "
            "refined is usually best for clean symmetry output."
        ),
    )
    parser.set_defaults(force_middle_li_to_ia3=True)
    parser.add_argument(
        "--force-middle-li-to-ia3",
        dest="force_middle_li_to_ia3",
        action="store_true",
        help=(
            "Try to map middle-Li structures to Ia-3 by scanning larger symprec values. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-force-middle-li-to-ia3",
        dest="force_middle_li_to_ia3",
        action="store_false",
        help="Disable Ia-3 targeting for middle-Li structures.",
    )
    parser.add_argument(
        "--ia3-scan-symprec",
        type=str,
        default=DEFAULT_IA3_SCAN,
        help=(
            "Comma-separated symprec list for Ia-3 scan, e.g. 0.01,0.03,0.05,0.1. "
            "The default --symprec is automatically included."
        ),
    )
    return parser.parse_args()


def expand_patterns(patterns: Iterable[str]) -> List[Path]:
    all_files: List[Path] = []
    for pattern in patterns:
        all_files.extend(Path(p) for p in glob(pattern, recursive=True))

    # Keep unique paths while preserving deterministic order.
    unique_sorted = sorted({p.resolve() for p in all_files})
    return unique_sorted


def sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def parse_symprec_scan(text: str, base_symprec: float) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid symprec value: {token}") from exc

        if value <= 0:
            continue
        if value not in values:
            values.append(value)

    if base_symprec > 0 and base_symprec not in values:
        values.insert(0, base_symprec)

    if not values:
        values = [0.01]

    return values


def get_li_count(structure) -> float:
    return float(structure.composition.get_el_amt_dict().get("Li", 0.0))


def get_li_fraction(structure) -> float:
    comp_dict = structure.composition.get_el_amt_dict()
    total = float(sum(comp_dict.values()))
    if total <= 0:
        return 0.0
    return float(comp_dict.get("Li", 0.0)) / total


def identify_middle_li_files(li_counts: Dict[Path, float]) -> Set[Path]:
    rounded_positive = sorted({round(v, 6) for v in li_counts.values() if v > 0})
    if len(rounded_positive) < 3:
        return set()

    min_li = rounded_positive[0]
    max_li = rounded_positive[-1]
    middle_files = {
        path
        for path, li_count in li_counts.items()
        if min_li < round(li_count, 6) < max_li
    }
    return middle_files


def normalize_sg_symbol(symbol: str) -> str:
    return symbol.replace(" ", "").replace("_", "").lower()


def is_ia3(symbol: str, number: int) -> bool:
    return number == IA3_NUMBER or normalize_sg_symbol(symbol) == "ia-3"


def find_ia3_with_scan(
    structure,
    angle_tolerance: float,
    symprec_scan: Iterable[float],
) -> Tuple[Optional[SpacegroupAnalyzer], Optional[float]]:
    for symprec in symprec_scan:
        sga = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        if is_ia3(sg_symbol, sg_number):
            return sga, symprec
    return None, None


def get_structure_for_export(sga: SpacegroupAnalyzer, mode: str):
    if mode == "refined":
        return sga.get_refined_structure()
    if mode == "conventional":
        return sga.get_conventional_standard_structure()
    if mode == "primitive":
        return sga.get_primitive_standard_structure()
    return sga.get_symmetrized_structure().structure


def main() -> None:
    args = parse_args()

    patterns = args.patterns if args.patterns else [DEFAULT_PATTERN]
    vasp_files = expand_patterns(patterns)

    try:
        ia3_symprec_scan = parse_symprec_scan(args.ia3_scan_symprec, args.symprec)
    except ValueError as exc:
        print(f"Invalid --ia3-scan-symprec: {exc}")
        return

    if not vasp_files:
        print("No input files found.")
        print("Checked patterns:")
        for p in patterns:
            print(f"  - {p}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    structures: Dict[Path, object] = {}
    preload_errors: Dict[Path, str] = {}
    li_counts: Dict[Path, float] = {}

    print(f"Found {len(vasp_files)} VASP file(s).")
    print("Preloading structures for Li concentration statistics...")
    for file_path in vasp_files:
        try:
            atoms = ase_read(str(file_path), format="vasp")
            structure = AseAtomsAdaptor.get_structure(atoms)
            structures[file_path] = structure
            li_counts[file_path] = get_li_count(structure)
        except Exception as exc:
            preload_errors[file_path] = f"{type(exc).__name__}: {exc}"
            li_counts[file_path] = 0.0

    middle_li_files = (
        identify_middle_li_files(li_counts) if args.force_middle_li_to_ia3 else set()
    )
    if args.force_middle_li_to_ia3:
        distinct_positive_li = sorted({round(v, 6) for v in li_counts.values() if v > 0})
        print(f"Distinct positive Li counts: {distinct_positive_li}")
        print(f"Middle-Li files targeted to {IA3_SYMBOL}: {len(middle_li_files)}")

    for file_path in vasp_files:
        row = {
            "input_file": str(file_path),
            "status": "failed",
            "li_count": "",
            "li_fraction": "",
            "spacegroup_symbol": "",
            "spacegroup_number": "",
            "crystal_system": "",
            "hall_symbol": "",
            "ia3_targeted": "no",
            "ia3_applied": "no",
            "used_symprec": "",
            "cif_output": "",
            "note": "",
            "error": "",
        }

        try:
            if file_path in preload_errors:
                raise RuntimeError(preload_errors[file_path])

            structure = structures[file_path]
            li_count = get_li_count(structure)
            li_fraction = get_li_fraction(structure)

            row["li_count"] = f"{li_count:.6f}".rstrip("0").rstrip(".") or "0"
            row["li_fraction"] = f"{li_fraction:.6f}"

            ia3_targeted = args.force_middle_li_to_ia3 and file_path in middle_li_files
            row["ia3_targeted"] = "yes" if ia3_targeted else "no"

            ia3_applied = False
            used_symprec = args.symprec

            if ia3_targeted:
                ia3_sga, matched_symprec = find_ia3_with_scan(
                    structure,
                    angle_tolerance=args.angle_tolerance,
                    symprec_scan=ia3_symprec_scan,
                )
                if ia3_sga is not None and matched_symprec is not None:
                    sga = ia3_sga
                    used_symprec = matched_symprec
                    ia3_applied = True
                    row["note"] = f"Matched {IA3_SYMBOL} in symprec scan."
                else:
                    sga = SpacegroupAnalyzer(
                        structure,
                        symprec=args.symprec,
                        angle_tolerance=args.angle_tolerance,
                    )
                    row["note"] = (
                        f"Targeted {IA3_SYMBOL} but not matched; fallback to default symprec."
                    )
            else:
                sga = SpacegroupAnalyzer(
                    structure,
                    symprec=args.symprec,
                    angle_tolerance=args.angle_tolerance,
                )

            row["ia3_applied"] = "yes" if ia3_applied else "no"
            row["used_symprec"] = f"{used_symprec:g}"

            sg_symbol = sga.get_space_group_symbol()
            sg_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            hall_symbol = sga.get_hall()

            export_structure = get_structure_for_export(sga, args.structure_mode)

            safe_sg = sanitize_for_filename(sg_symbol)
            if ia3_targeted:
                target_tag = "Ia3" if ia3_applied else "Ia3Fallback"
                cif_name = f"{file_path.stem}_{target_tag}_SG{sg_number}_{safe_sg}.cif"
            else:
                cif_name = f"{file_path.stem}_SG{sg_number}_{safe_sg}.cif"
            cif_path = out_dir / cif_name

            # symprec in CifWriter writes symmetry operators/space-group info when available.
            writer = CifWriter(
                export_structure,
                symprec=used_symprec,
                angle_tolerance=args.angle_tolerance,
            )
            writer.write_file(str(cif_path))

            row.update(
                {
                    "status": "ok",
                    "spacegroup_symbol": sg_symbol,
                    "spacegroup_number": str(sg_number),
                    "crystal_system": crystal_system,
                    "hall_symbol": hall_symbol,
                    "cif_output": str(cif_path),
                }
            )

            ia3_msg = ""
            if ia3_targeted:
                ia3_msg = (
                    f" [target {IA3_SYMBOL}: {'matched' if ia3_applied else 'fallback'}]"
                )

            print(
                f"[OK] {file_path.name} -> SG {sg_number} ({sg_symbol}) "
                f"[symprec={used_symprec:g}] -> {cif_path.name}{ia3_msg}"
            )

        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {file_path.name} -> {row['error']}")

        summary_rows.append(row)

    summary_csv = out_dir / "symmetry_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_file",
                "status",
                "li_count",
                "li_fraction",
                "spacegroup_symbol",
                "spacegroup_number",
                "crystal_system",
                "hall_symbol",
                "ia3_targeted",
                "ia3_applied",
                "used_symprec",
                "cif_output",
                "note",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    ok_count = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_count = len(summary_rows) - ok_count
    ia3_target_count = sum(1 for r in summary_rows if r["ia3_targeted"] == "yes")
    ia3_applied_count = sum(1 for r in summary_rows if r["ia3_applied"] == "yes")

    print("\nDone.")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print(f"Success: {ok_count}, Failed: {fail_count}")
    if args.force_middle_li_to_ia3:
        print(f"Ia-3 targeted: {ia3_target_count}, Ia-3 matched: {ia3_applied_count}")


if __name__ == "__main__":
    main()
