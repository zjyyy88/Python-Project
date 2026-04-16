#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser
    from pymatgen.io.vasp import Poscar
except Exception as exc:  # noqa: BLE001
    print("[ERROR] pymatgen import failed. Please install it first: pip install pymatgen")
    print(f"[DETAIL] {exc}")
    sys.exit(1)


DEFAULT_ROOT = r"E:\固态组\LiLa2O3\La2O3-Lithiation\wz-SEND-convex"


def li_folder_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"Li(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def read_cif_structure(cif_path: Path) -> Structure:
    parser = CifParser(str(cif_path), occupancy_tolerance=100)
    if hasattr(parser, "parse_structures"):
        structures = parser.parse_structures(primitive=False)
    else:
        structures = parser.get_structures(primitive=False)

    if not structures:
        raise ValueError("No structure parsed from CIF")
    return structures[0]


def remove_zero_occupancy_species(
    structure: Structure,
    zero_occ_threshold: float,
) -> tuple[Structure, int]:
    species: list[object] = []
    frac_coords: list[object] = []
    removed_empty_sites = 0

    for site in structure.sites:
        kept = [(sp, occ) for sp, occ in site.species.items() if occ > zero_occ_threshold]
        if not kept:
            removed_empty_sites += 1
            continue

        # VASP format cannot represent partial occupancy directly.
        # Keep the highest-occupancy species for each site after filtering zeros.
        chosen_species, _ = max(kept, key=lambda item: item[1])
        species.append(chosen_species)
        frac_coords.append(site.frac_coords)

    cleaned_structure = Structure(
        lattice=structure.lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )
    return cleaned_structure, removed_empty_sites


def convert_li_folders(
    root: Path,
    li_pattern: str = "Li*",
    zero_occ_threshold: float = 1e-8,
) -> int:
    li_dirs = sorted([p for p in root.glob(li_pattern) if p.is_dir()], key=li_folder_sort_key)
    if not li_dirs:
        print(f"[ERROR] No Li* folders found under: {root}")
        return 1

    total_converted = 0
    for li_dir in li_dirs:
        cif_files = sorted(li_dir.glob("*.cif"))
        if not cif_files:
            print(f"[WARN] Skip {li_dir}: no .cif file")
            continue

        print(f"[INFO] Enter folder: {li_dir}")
        for cif_path in cif_files:
            vasp_path = cif_path.with_suffix(".vasp")
            try:
                structure = read_cif_structure(cif_path)
                cleaned_structure, removed_sites = remove_zero_occupancy_species(
                    structure,
                    zero_occ_threshold=zero_occ_threshold,
                )
                Poscar(cleaned_structure).write_file(str(vasp_path))
                total_converted += 1
                print(
                    f"  [OK] {cif_path.name} -> {vasp_path.name} "
                    f"(removed empty sites: {removed_sites})"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {cif_path}: {exc}")

    print(f"[DONE] Converted {total_converted} cif file(s).")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert *.cif to *.vasp in Li* folders using pymatgen."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Root directory containing Li* folders.",
    )
    parser.add_argument(
        "--li-pattern",
        default="Li*",
        help="Folder glob pattern under root, default: Li*",
    )
    parser.add_argument(
        "--zero-occ-threshold",
        type=float,
        default=1e-8,
        help="Treat occupancy <= threshold as zero and remove the species/site.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"[ERROR] Root path not found: {root}")
        return 1

    return convert_li_folders(
        root,
        li_pattern=args.li_pattern,
        zero_occ_threshold=args.zero_occ_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
