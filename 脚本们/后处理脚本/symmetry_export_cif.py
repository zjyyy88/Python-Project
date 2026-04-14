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
from typing import Iterable, List

from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

DEFAULT_PATTERN = r"E:\固态组\LiLa2O3\LixMgxLa32-xO48\掺杂base结构\*.vasp"


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

    if not vasp_files:
        print("No input files found.")
        print("Checked patterns:")
        for p in patterns:
            print(f"  - {p}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"Found {len(vasp_files)} VASP file(s).")
    for file_path in vasp_files:
        row = {
            "input_file": str(file_path),
            "status": "failed",
            "spacegroup_symbol": "",
            "spacegroup_number": "",
            "crystal_system": "",
            "hall_symbol": "",
            "cif_output": "",
            "error": "",
        }

        try:
            atoms = ase_read(str(file_path), format="vasp")
            structure = AseAtomsAdaptor.get_structure(atoms)

            sga = SpacegroupAnalyzer(
                structure,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
            )

            sg_symbol = sga.get_space_group_symbol()
            sg_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            hall_symbol = sga.get_hall()

            export_structure = get_structure_for_export(sga, args.structure_mode)

            safe_sg = sanitize_for_filename(sg_symbol)
            cif_name = f"{file_path.stem}_SG{sg_number}_{safe_sg}.cif"
            cif_path = out_dir / cif_name

            # symprec in CifWriter writes symmetry operators/space-group info when available.
            writer = CifWriter(
                export_structure,
                symprec=args.symprec,
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

            print(
                f"[OK] {file_path.name} -> SG {sg_number} ({sg_symbol}) -> {cif_path.name}"
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
                "spacegroup_symbol",
                "spacegroup_number",
                "crystal_system",
                "hall_symbol",
                "cif_output",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    ok_count = sum(1 for r in summary_rows if r["status"] == "ok")
    fail_count = len(summary_rows) - ok_count

    print("\nDone.")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print(f"Success: {ok_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
