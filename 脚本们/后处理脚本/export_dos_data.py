"""
从 VASP 的 vasprun.xml 导出 DOS 数据（仅导出 CSV，不画图）。

功能
- 可导出类型：tdos、element、element_spd、element_orbital、site、site_spd、site_orbital。
- 支持两种选择方式：
    1) 交互模式（类似 VASPKIT 的提示选择）
    2) 命令行参数模式（适合批处理自动化）
- 默认能量轴为 E-Efermi；加 --absolute-energy 可导出绝对能量。
- 可导出 p、d 的细分轨道：px/py/pz、dxy/dxz/dyz/dz2/dx2-y2。
- 会同时生成 site_index_map.txt，方便按原子序号选择 site。

快速使用
1) 交互模式（推荐手动挑选）
        python export_dos_data.py --vasprun "D:/path/to/vasprun.xml" --interactive

2) 导出全部类别
        python export_dos_data.py --vasprun "D:/path/to/vasprun.xml" --exports all --elements all --sites all --index-base 1

3) 仅导出指定项目
        python export_dos_data.py --vasprun "D:/path/to/vasprun.xml" --exports element site_spd --elements In Cl --sites 1 2 8-12 --index-base 1

输出文件
- 默认输出目录：<vasprun_dir>/dos_data（也可用 --outdir 自定义）
- 主要输出：
    - 各类 DOS 的 CSV 文件
    - site_index_map.txt
    - export_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos
from pymatgen.io.vasp import Vasprun

EXPORT_TYPES = (
    "tdos",
    "element",
    "element_spd",
    "element_orbital",
    "site",
    "site_spd",
    "site_orbital",
)
TARGET_ORBITAL_NAMES = ("px", "py", "pz", "dxy", "dxz", "dyz", "dz2", "dx2")


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def unique_keep_order(items: Iterable[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_int_tokens(tokens: Sequence[str]) -> list[int] | None:
    values: list[int] = []
    for token in tokens:
        for part in re.split(r"[\s,]+", token.strip()):
            if not part:
                continue
            lower = part.lower()
            if lower == "all":
                return None
            if "-" in part:
                seg = part.split("-", 1)
                if len(seg) != 2 or not seg[0] or not seg[1]:
                    raise ValueError(f"Invalid range token: {part}")
                start = int(seg[0])
                end = int(seg[1])
                step = 1 if end >= start else -1
                values.extend(range(start, end + step, step))
            else:
                values.append(int(part))
    return unique_keep_order(values)


def user_index_list(n_sites: int, index_base: int) -> list[int]:
    if index_base == 1:
        return list(range(1, n_sites + 1))
    return list(range(0, n_sites))


def user_to_internal_index(user_index: int, index_base: int) -> int:
    if index_base == 1:
        return user_index - 1
    return user_index


def has_spin_down(dos) -> bool:
    return Spin.down in dos.densities and dos.densities[Spin.down] is not None


def get_spin_up_density(dos):
    if Spin.up in dos.densities:
        return dos.densities[Spin.up]
    return next(iter(dos.densities.values()))


def normalize_orbital_output_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in {"dx2", "dx2_y2"}:
        return "dx2_y2"
    return key


def get_component_orbitals() -> list[tuple[str, Orbital]]:
    out: list[tuple[str, Orbital]] = []
    for name in TARGET_ORBITAL_NAMES:
        if hasattr(Orbital, name):
            out.append((normalize_orbital_output_name(name), getattr(Orbital, name)))
        else:
            print(f"[Warning] Orbital enum not available and skipped: {name}")
    return out


def sum_dos_objects(dos_list: Sequence[Dos]) -> Dos:
    if not dos_list:
        raise ValueError("dos_list is empty")

    base = dos_list[0]
    densities = {
        spin: np.array(values, dtype=float)
        for spin, values in base.densities.items()
    }
    for dos_obj in dos_list[1:]:
        for spin, values in dos_obj.densities.items():
            values_arr = np.array(values, dtype=float)
            if spin in densities:
                densities[spin] += values_arr
            else:
                densities[spin] = values_arr

    return Dos(efermi=base.efermi, energies=base.energies, densities=densities)


def site_has_element(site, symbol: str) -> bool:
    symbol_upper = symbol.upper()
    for specie in site.species.keys():
        if getattr(specie, "symbol", str(specie)).upper() == symbol_upper:
            return True
    return False


def write_dos_csv(dos, out_csv: Path, zero_at_efermi: bool) -> None:
    energies = dos.energies - dos.efermi if zero_at_efermi else dos.energies
    up = get_spin_up_density(dos)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if has_spin_down(dos):
            down = dos.densities[Spin.down]
            writer.writerow(["energy_eV", "dos_up", "dos_down"])
            for e, d_up, d_dn in zip(energies, up, down):
                writer.writerow([f"{float(e):.8f}", f"{float(d_up):.12e}", f"{float(d_dn):.12e}"])
        else:
            writer.writerow(["energy_eV", "dos"])
            for e, d_up in zip(energies, up):
                writer.writerow([f"{float(e):.8f}", f"{float(d_up):.12e}"])


def write_site_index_map(out_file: Path, structure, index_base: int) -> None:
    header = [
        "# Site index map",
        f"# INDEX_BASE = {index_base}",
        f"# Total sites = {len(structure)}",
        "index\tspecies\tfrac_coords",
    ]
    lines = []
    for i, site in enumerate(structure):
        idx_show = i + 1 if index_base == 1 else i
        frac = ", ".join(f"{v:.6f}" for v in site.frac_coords)
        lines.append(f"{idx_show}\t{site.species_string}\t{frac}")
    out_file.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def choose_exports_interactive() -> set[str]:
    print("\nSelect export type(s):")
    print("  1) tdos       - total DOS")
    print("  2) element    - element-resolved DOS")
    print("  3) element_spd- element s/p/d DOS")
    print("  4) site       - selected site LDOS")
    print("  5) site_spd   - selected site s/p/d DOS")
    print("  6) element_orbital - element px/py/pz + five d orbitals")
    print("  7) site_orbital    - site px/py/pz + five d orbitals")
    print("  0) all")
    raw = input("Enter numbers (e.g. 1 2 5) or names: ").strip()
    if not raw:
        return {"tdos"}

    mapping = {
        "0": "all",
        "1": "tdos",
        "2": "element",
        "3": "element_spd",
        "4": "site",
        "5": "site_spd",
        "6": "element_orbital",
        "7": "site_orbital",
    }
    selected: set[str] = set()
    for token in re.split(r"[\s,]+", raw):
        if not token:
            continue
        key = token.lower()
        mapped = mapping.get(key, key)
        if mapped == "all":
            return set(EXPORT_TYPES)
        if mapped in EXPORT_TYPES:
            selected.add(mapped)
        else:
            print(f"[Warning] Unknown export token ignored: {token}")

    if not selected:
        selected.add("tdos")
    return selected


def choose_elements_interactive(available_symbols: Sequence[str]) -> list[str]:
    print("\nAvailable elements:", " ".join(available_symbols))
    raw = input("Select elements (e.g. In Cl O) or all [default=all]: ").strip()
    if not raw:
        return list(available_symbols)

    requested = [x for x in re.split(r"[\s,]+", raw) if x]
    if any(x.lower() == "all" for x in requested):
        return list(available_symbols)

    avail_upper = {s.upper(): s for s in available_symbols}
    selected: list[str] = []
    for token in requested:
        key = token.upper()
        if key in avail_upper:
            selected.append(avail_upper[key])
        else:
            print(f"[Warning] Element not found and ignored: {token}")
    return list(dict.fromkeys(selected))


def choose_sites_interactive(n_sites: int, index_base: int) -> list[int]:
    print("\nSite selection examples: 1 2 8-12   or   all")
    raw = input(
        f"Select site indices (INDEX_BASE={index_base}, total={n_sites}) [default=all]: "
    ).strip()
    if not raw:
        return user_index_list(n_sites, index_base)

    parsed = parse_int_tokens([raw])
    if parsed is None:
        return user_index_list(n_sites, index_base)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DOS data from vasprun.xml to CSV files without plotting."
    )
    parser.add_argument("--vasprun", type=Path, required=True, help="Path to vasprun.xml")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <vasprun_dir>/dos_data)",
    )
    parser.add_argument(
        "--exports",
        nargs="+",
        default=["tdos"],
        choices=[*EXPORT_TYPES, "all"],
        help="Export items to generate",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=[],
        help="Element symbols for element/element_spd/element_orbital exports (e.g. In Cl O or all)",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=[],
        help="Site indices for site/site_spd/site_orbital exports (e.g. 1 2 8-12 or all)",
    )
    parser.add_argument(
        "--index-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Site indexing convention: 1 (VESTA style) or 0 (Python style)",
    )
    parser.add_argument(
        "--absolute-energy",
        action="store_true",
        help="Use absolute energy instead of E-Efermi",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (VASPKIT-like selection prompts)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vasprun_path = args.vasprun
    if not vasprun_path.exists():
        raise FileNotFoundError(f"vasprun.xml not found: {vasprun_path}")

    outdir = args.outdir if args.outdir else vasprun_path.parent / "dos_data"
    outdir.mkdir(parents=True, exist_ok=True)

    zero_at_efermi = not args.absolute_energy

    vrun = Vasprun(str(vasprun_path), parse_potcar_file=False)
    complete_dos = vrun.complete_dos
    structure = complete_dos.structure

    write_site_index_map(outdir / "site_index_map.txt", structure, args.index_base)

    if args.interactive:
        exports = choose_exports_interactive()
    else:
        exports = set(args.exports)
        if "all" in exports:
            exports = set(EXPORT_TYPES)

    written_files: list[Path] = []

    if "tdos" in exports:
        out_file = outdir / "tdos.csv"
        write_dos_csv(vrun.tdos, out_file, zero_at_efermi)
        written_files.append(out_file)

    element_dos = complete_dos.get_element_dos()
    symbol_to_element = {str(ele): ele for ele in element_dos}
    available_symbols = sorted(symbol_to_element.keys())

    selected_elements: list[str] = []
    if "element" in exports or "element_spd" in exports or "element_orbital" in exports:
        if args.interactive:
            selected_elements = choose_elements_interactive(available_symbols)
        else:
            if not args.elements or any(x.lower() == "all" for x in args.elements):
                selected_elements = available_symbols
            else:
                upper_to_symbol = {s.upper(): s for s in available_symbols}
                for token in args.elements:
                    key = token.upper()
                    if key in upper_to_symbol:
                        selected_elements.append(upper_to_symbol[key])
                    else:
                        print(f"[Warning] Element not found and ignored: {token}")
                selected_elements = list(dict.fromkeys(selected_elements))

    if "element" in exports:
        for symbol in selected_elements:
            ele_obj = symbol_to_element[symbol]
            out_file = outdir / f"element_{safe_name(symbol)}.csv"
            write_dos_csv(element_dos[ele_obj], out_file, zero_at_efermi)
            written_files.append(out_file)

    if "element_spd" in exports:
        for symbol in selected_elements:
            ele_obj = symbol_to_element[symbol]
            spd_dict = complete_dos.get_element_spd_dos(ele_obj)
            for orb, dos_obj in spd_dict.items():
                orb_name = safe_name(getattr(orb, "name", str(orb)).lower())
                out_file = outdir / f"element_{safe_name(symbol)}_{orb_name}.csv"
                write_dos_csv(dos_obj, out_file, zero_at_efermi)
                written_files.append(out_file)

    if "element_orbital" in exports:
        component_orbitals = get_component_orbitals()
        for symbol in selected_elements:
            matched_sites = [site for site in structure if site_has_element(site, symbol)]
            if not matched_sites:
                print(f"[Warning] No matched sites for element: {symbol}")
                continue

            for orb_file_name, orbital in component_orbitals:
                dos_parts: list[Dos] = []
                for site in matched_sites:
                    try:
                        dos_parts.append(complete_dos.get_site_orbital_dos(site, orbital))
                    except Exception:
                        continue

                if not dos_parts:
                    continue

                summed = sum_dos_objects(dos_parts)
                out_file = outdir / f"element_{safe_name(symbol)}_{orb_file_name}.csv"
                write_dos_csv(summed, out_file, zero_at_efermi)
                written_files.append(out_file)

    selected_sites: list[int] = []
    if "site" in exports or "site_spd" in exports or "site_orbital" in exports:
        n_sites = len(structure)
        if args.interactive:
            selected_sites = choose_sites_interactive(n_sites, args.index_base)
        else:
            if not args.sites:
                print("[Info] --sites not provided. Exporting all sites for site-related outputs.")
                selected_sites = user_index_list(n_sites, args.index_base)
            else:
                parsed = parse_int_tokens(args.sites)
                selected_sites = (
                    user_index_list(n_sites, args.index_base) if parsed is None else parsed
                )

        valid_sites: list[int] = []
        for user_idx in selected_sites:
            i0 = user_to_internal_index(user_idx, args.index_base)
            if 0 <= i0 < len(structure):
                valid_sites.append(user_idx)
            else:
                print(
                    f"[Warning] Site index out of range and ignored: {user_idx} "
                    f"(INDEX_BASE={args.index_base}, total={len(structure)})"
                )
        selected_sites = unique_keep_order(valid_sites)

    if "site" in exports:
        for user_idx in selected_sites:
            i0 = user_to_internal_index(user_idx, args.index_base)
            site = structure[i0]
            tag = safe_name(f"site_{user_idx}_{site.species_string}")
            out_file = outdir / f"{tag}.csv"
            write_dos_csv(complete_dos.get_site_dos(site), out_file, zero_at_efermi)
            written_files.append(out_file)

    if "site_spd" in exports:
        for user_idx in selected_sites:
            i0 = user_to_internal_index(user_idx, args.index_base)
            site = structure[i0]
            tag = safe_name(f"site_{user_idx}_{site.species_string}")
            spd_dict = complete_dos.get_site_spd_dos(site)
            for orb, dos_obj in spd_dict.items():
                orb_name = safe_name(getattr(orb, "name", str(orb)).lower())
                out_file = outdir / f"{tag}_{orb_name}.csv"
                write_dos_csv(dos_obj, out_file, zero_at_efermi)
                written_files.append(out_file)

    if "site_orbital" in exports:
        component_orbitals = get_component_orbitals()
        for user_idx in selected_sites:
            i0 = user_to_internal_index(user_idx, args.index_base)
            site = structure[i0]
            tag = safe_name(f"site_{user_idx}_{site.species_string}")

            for orb_file_name, orbital in component_orbitals:
                try:
                    dos_obj = complete_dos.get_site_orbital_dos(site, orbital)
                except Exception:
                    continue

                out_file = outdir / f"{tag}_{orb_file_name}.csv"
                write_dos_csv(dos_obj, out_file, zero_at_efermi)
                written_files.append(out_file)

    info_lines = [
        f"vasprun: {vasprun_path}",
        f"efermi: {vrun.efermi}",
        f"zero_at_efermi: {zero_at_efermi}",
        f"index_base: {args.index_base}",
        "exports: " + ", ".join(sorted(exports)),
        f"n_files: {len(written_files)}",
    ]
    (outdir / "export_summary.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    print(f"Done. Wrote {len(written_files)} DOS file(s) to: {outdir}")
    print(f"Site index map: {outdir / 'site_index_map.txt'}")


if __name__ == "__main__":
    main()
