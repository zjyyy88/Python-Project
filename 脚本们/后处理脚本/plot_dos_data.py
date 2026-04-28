"""
从 dos_data 目录读取 CSV 并绘制 DOS 图。

支持两类输入：
1) 元素模式（element）
   - total: 不同元素的 total DOS
    - spd: 不同元素的分波 DOS（支持 s/p/d，也支持 px/py/pz 与 5 个 d 轨道）
   - both: total + spd 都画
2) 原子模式（site）
    - spd: 不同序号原子的分波 DOS（支持 s/p/d，也支持 px/py/pz 与 5 个 d 轨道）

可选绘图行为：
- SPD 是否画在同一张图（--spd-together / --spd-separate）
- 不同元素或不同原子是否做同横坐标组图（--group-subplots / --no-group-subplots）

默认数据目录即你当前项目数据目录：
D:\aaazjy\2025.4.1\InCl3·3H2O\case\reaction\newVCl2OH-27712-USPEX-InCl2OH\dos_data

示例：
1) 元素模式，画 In/Cl 的 total+spd，SPD 同图，组图输出
python plot_dos_data.py --mode element --element-kind both --elements In Cl --group-subplots --spd-together

2) 原子模式，画 1 2 8-10 号原子的 spd，SPD 拆分成单图，组图输出
python plot_dos_data.py --mode site --sites 1 2 8-10 --spd-separate --group-subplots

3) 交互模式（一步步输入）
python plot_dos_data.py --interactive
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_DIR = Path(
    r"D:\aaazjy\2025.4.1\YCl3·6H2O\convex\1H2O-2\dos_data"
)
ORBITAL_ORDER = [
    "s",
    "p",
    "px",
    "py",
    "pz",
    "d",
    "dxy",
    "dxz",
    "dyz",
    "dz2",
    "dx2_y2",
    "f",
    "g",
]
ORBITAL_COLORS = {
    "s": "#1f77b4",
    "p": "#ff7f0e",
    "px": "#f28e2b",
    "py": "#e15759",
    "pz": "#b07aa1",
    "d": "#2ca02c",
    "dxy": "#59a14f",
    "dxz": "#4e79a7",
    "dyz": "#76b7b2",
    "dz2": "#8cd17d",
    "dx2_y2": "#499894",
    "f": "#d62728",
    "g": "#9467bd",
}
ORBITAL_TOKEN_SET = set(ORBITAL_ORDER) | {"dx2"}


@dataclass
class DosIndex:
    element_total: dict[str, Path]
    element_spd: dict[str, dict[str, Path]]
    site_total: dict[int, Path]
    site_spd: dict[int, dict[str, Path]]
    site_species: dict[int, str]


@dataclass
class CurveSpec:
    label: str
    path: Path


@dataclass
class PanelSpec:
    title: str
    curves: list[CurveSpec]
    file_stem: str


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def normalize_orbital_label(label: str) -> str:
    token = label.lower().replace("-", "_")
    if token in {"dx2", "dx2_y2"}:
        return "dx2_y2"
    return token


def display_orbital_label(label: str) -> str:
    token = normalize_orbital_label(label)
    if token == "dx2_y2":
        return "dx2-y2"
    return token


def split_tokens(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        for part in re.split(r"[\s,]+", token.strip()):
            if part:
                out.append(part)
    return out


def split_label_orbital(token: str) -> tuple[str, str | None]:
    if "_" not in token:
        return token, None

    left, right = token.rsplit("_", 1)
    right_norm = normalize_orbital_label(right)
    if right_norm in ORBITAL_TOKEN_SET:
        return left, right_norm
    return token, None


def unique_keep_order(items: Sequence[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_index_tokens(tokens: Sequence[str]) -> list[int] | None:
    values: list[int] = []
    for token in split_tokens(tokens):
        if token.lower() == "all":
            return None
        if "-" in token:
            left, right = token.split("-", 1)
            if not left or not right:
                raise ValueError(f"无效区间: {token}")
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    return unique_keep_order(values)


def read_dos_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    energy: list[float] = []
    dos: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = float(row["energy_eV"])
            if "dos" in row and row["dos"] not in ("", None):
                y = float(row["dos"])
            elif (
                "dos_up" in row
                and "dos_down" in row
                and row["dos_up"] not in ("", None)
                and row["dos_down"] not in ("", None)
            ):
                y = float(row["dos_up"]) + float(row["dos_down"])
            else:
                raise ValueError(f"无法识别 DOS 列: {path}")
            energy.append(e)
            dos.append(y)

    return np.array(energy), np.array(dos)


def scan_dos_data(data_dir: Path) -> DosIndex:
    element_total: dict[str, Path] = {}
    element_spd: dict[str, dict[str, Path]] = {}
    site_total: dict[int, Path] = {}
    site_spd: dict[int, dict[str, Path]] = {}
    site_species: dict[int, str] = {}

    for csv_path in data_dir.glob("*.csv"):
        stem = csv_path.stem

        if stem.startswith("element_"):
            body = stem[len("element_") :]
            symbol, orb = split_label_orbital(body)
            if orb is None:
                element_total[symbol] = csv_path
            else:
                element_spd.setdefault(symbol, {})[orb] = csv_path
            continue

        if stem.startswith("site_"):
            body = stem[len("site_") :]
            idx_raw, sep, tail = body.partition("_")
            if not sep or not idx_raw.isdigit() or not tail:
                continue

            idx = int(idx_raw)
            species, orb = split_label_orbital(tail)
            site_species[idx] = species
            if orb is None:
                site_total[idx] = csv_path
            else:
                site_spd.setdefault(idx, {})[orb] = csv_path

    return DosIndex(
        element_total=element_total,
        element_spd=element_spd,
        site_total=site_total,
        site_spd=site_spd,
        site_species=site_species,
    )


def parse_elements(tokens: Sequence[str], available: Sequence[str]) -> list[str]:
    available_map = {s.upper(): s for s in available}
    req = split_tokens(tokens)
    if not req or any(x.lower() == "all" for x in req):
        return list(available)

    selected: list[str] = []
    for token in req:
        key = token.upper()
        if key in available_map:
            selected.append(available_map[key])
        else:
            print(f"[Warning] 未找到元素，已忽略: {token}")
    return list(dict.fromkeys(selected))


def parse_sites(tokens: Sequence[str], available: Sequence[int]) -> list[int]:
    parsed = parse_index_tokens(tokens)
    if parsed is None:
        return list(available)

    avail_set = set(available)
    selected = [x for x in parsed if x in avail_set]
    missing = [x for x in parsed if x not in avail_set]
    for x in missing:
        print(f"[Warning] 未找到原子序号对应 DOS 数据，已忽略: {x}")
    return selected


def ordered_orbitals(d: dict[str, Path]) -> list[str]:
    known = [x for x in ORBITAL_ORDER if x in d]
    other = sorted(x for x in d.keys() if x not in ORBITAL_ORDER)
    return known + other


def build_element_total_panels(index: DosIndex, elements: list[str]) -> list[PanelSpec]:
    panels: list[PanelSpec] = []
    for symbol in elements:
        path = index.element_total.get(symbol)
        if path is None:
            print(f"[Warning] 元素 total 数据缺失: {symbol}")
            continue
        panels.append(
            PanelSpec(
                title=f"Element {symbol} total DOS",
                curves=[CurveSpec(label=symbol, path=path)],
                file_stem=f"element_{safe_name(symbol)}_total",
            )
        )
    return panels


def build_element_spd_panels(
    index: DosIndex,
    elements: list[str],
    spd_together: bool,
) -> list[PanelSpec]:
    panels: list[PanelSpec] = []
    for symbol in elements:
        spd = index.element_spd.get(symbol, {})
        orbs = ordered_orbitals(spd)
        if not orbs:
            print(f"[Warning] 元素 SPD 数据缺失: {symbol}")
            continue

        if spd_together:
            curves = [CurveSpec(label=display_orbital_label(orb), path=spd[orb]) for orb in orbs]
            panels.append(
                PanelSpec(
                    title=f"Element {symbol} SPD DOS",
                    curves=curves,
                    file_stem=f"element_{safe_name(symbol)}_spd",
                )
            )
        else:
            for orb in orbs:
                label = display_orbital_label(orb)
                panels.append(
                    PanelSpec(
                        title=f"Element {symbol} {label}-DOS",
                        curves=[CurveSpec(label=label, path=spd[orb])],
                        file_stem=f"element_{safe_name(symbol)}_{safe_name(orb)}",
                    )
                )
    return panels


def build_site_spd_panels(index: DosIndex, sites: list[int], spd_together: bool) -> list[PanelSpec]:
    panels: list[PanelSpec] = []
    for idx in sites:
        spd = index.site_spd.get(idx, {})
        species = index.site_species.get(idx, "X")
        orbs = ordered_orbitals(spd)
        if not orbs:
            print(f"[Warning] 原子 {idx} 的 SPD 数据缺失")
            continue

        tag = safe_name(f"site_{idx}_{species}")
        if spd_together:
            curves = [CurveSpec(label=display_orbital_label(orb), path=spd[orb]) for orb in orbs]
            panels.append(
                PanelSpec(
                    title=f"Site {idx} ({species}) SPD DOS",
                    curves=curves,
                    file_stem=f"{tag}_spd",
                )
            )
        else:
            for orb in orbs:
                label = display_orbital_label(orb)
                panels.append(
                    PanelSpec(
                        title=f"Site {idx} ({species}) {label}-DOS",
                        curves=[CurveSpec(label=label, path=spd[orb])],
                        file_stem=f"{tag}_{safe_name(orb)}",
                    )
                )
    return panels


def get_curve_data(
    path: Path,
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if path not in cache:
        cache[path] = read_dos_csv(path)
    return cache[path]


def apply_axis_style(ax, xlim: tuple[float, float] | None) -> None:
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_ylabel("DOS")
    if xlim is not None:
        ax.set_xlim(*xlim)


def render_panels(
    panels: list[PanelSpec],
    outdir: Path,
    group_subplots: bool,
    xlim: tuple[float, float] | None,
    dpi: int,
    show: bool,
    group_name: str,
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> list[Path]:
    out_files: list[Path] = []
    if not panels:
        return out_files

    if group_subplots:
        n = len(panels)
        #ncols = 2 if n <= 4 else 3
        ncols = 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.0 * ncols, 3.8 * nrows),
            sharex=True,
            squeeze=False,
        )

        for i, panel in enumerate(panels):
            ax = axes[i // ncols][i % ncols]
            for curve in panel.curves:
                x, y = get_curve_data(curve.path, cache)
                color = ORBITAL_COLORS.get(normalize_orbital_label(curve.label))
                ax.plot(x, y, label=curve.label, linewidth=1.5, color=color)
            apply_axis_style(ax, xlim)
            ax.set_title(panel.title, fontsize=10)
            if len(panel.curves) > 1:
                ax.legend(frameon=False, fontsize=9)

        # 清理多余子图
        for j in range(len(panels), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        for c in range(ncols):
            axes[nrows - 1][c].set_xlabel("Energy (eV)")

        fig.tight_layout()
        out_path = outdir / f"{safe_name(group_name)}_group.png"
        fig.savefig(out_path, dpi=dpi)
        out_files.append(out_path)
        if show:
            plt.show()
        plt.close(fig)
        return out_files

    # 非组图：每个 panel 输出单独图片
    for panel in panels:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for curve in panel.curves:
            x, y = get_curve_data(curve.path, cache)
            color = ORBITAL_COLORS.get(normalize_orbital_label(curve.label))
            ax.plot(x, y, label=curve.label, linewidth=1.6, color=color)

        apply_axis_style(ax, xlim)
        ax.set_title(panel.title)
        ax.set_xlabel("Energy (eV)")
        if len(panel.curves) > 1:
            ax.legend(frameon=False)
        fig.tight_layout()

        out_path = outdir / f"{panel.file_stem}.png"
        fig.savefig(out_path, dpi=dpi)
        out_files.append(out_path)

        if show:
            plt.show()
        plt.close(fig)

    return out_files


def ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def ask_xlim() -> tuple[float, float] | None:
    raw = input("是否设置 x 轴范围？例如 -8 8（直接回车表示自动）: ").strip()
    if not raw:
        return None
    parts = split_tokens([raw])
    if len(parts) != 2:
        print("[Warning] xlim 输入无效，改为自动")
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("[Warning] xlim 输入无效，改为自动")
        return None


def interactive_pick(
    index: DosIndex,
) -> tuple[str, str, list[str], list[int], bool, bool, tuple[float, float] | None]:
    mode_raw = input("选择模式：element 或 site（默认 element）: ").strip().lower()
    mode = mode_raw if mode_raw in {"element", "site"} else "element"

    element_kind = "spd"
    selected_elements: list[str] = []
    selected_sites: list[int] = []

    if mode == "element":
        kind_raw = input("元素模式选择：total / spd / both（默认 both）: ").strip().lower()
        element_kind = kind_raw if kind_raw in {"total", "spd", "both"} else "both"

        available_elements = sorted(
            set(index.element_total.keys()) | set(index.element_spd.keys())
        )
        print("可用元素:", " ".join(available_elements))
        ele_raw = input("输入元素（例如 In Cl，或 all；默认 all）: ").strip()
        selected_elements = parse_elements([ele_raw] if ele_raw else ["all"], available_elements)

    else:
        available_sites = sorted(index.site_spd.keys())
        print("可用原子序号:", " ".join(str(x) for x in available_sites))
        site_raw = input("输入原子序号（例如 1 2 8-10，或 all；默认 all）: ").strip()
        selected_sites = parse_sites([site_raw] if site_raw else ["all"], available_sites)

    spd_together = ask_yes_no("SPD 是否画在同一张图中", True)
    group_subplots = ask_yes_no("是否将不同元素/原子做成同横坐标组图", True)
    xlim = ask_xlim()

    return mode, element_kind, selected_elements, selected_sites, spd_together, group_subplots, xlim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 dos_data CSV 并绘图")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dos_data 目录路径",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="图片输出目录（默认: <data-dir>/plots）",
    )
    parser.add_argument(
        "--mode",
        choices=["element", "site"],
        default="element",
        help="绘图模式：元素或原子",
    )
    parser.add_argument(
        "--element-kind",
        choices=["total", "spd", "both"],
        default="both",
        help="元素模式下绘图内容",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=["all"],
        help="元素列表，例如 In Cl O 或 all",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=["all"],
        help="原子序号列表，例如 1 2 8-10 或 all",
    )

    spd_group = parser.add_mutually_exclusive_group()
    spd_group.add_argument(
        "--spd-together",
        dest="spd_together",
        action="store_true",
        help="SPD 画在同一张图（默认）",
    )
    spd_group.add_argument(
        "--spd-separate",
        dest="spd_together",
        action="store_false",
        help="SPD 拆分成单独图",
    )
    parser.set_defaults(spd_together=True)

    group_mode = parser.add_mutually_exclusive_group()
    group_mode.add_argument(
        "--group-subplots",
        dest="group_subplots",
        action="store_true",
        help="不同元素/原子输出为组图（同横坐标）",
    )
    group_mode.add_argument(
        "--no-group-subplots",
        dest="group_subplots",
        action="store_false",
        help="不同元素/原子输出为多张单图",
    )
    parser.set_defaults(group_subplots=True)

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="x 轴范围，例如 --xlim -8 8",
    )
    parser.add_argument("--dpi", type=int, default=300, help="输出图片 DPI")
    parser.add_argument("--show", action="store_true", help="保存后弹出显示")
    parser.add_argument("--interactive", action="store_true", help="交互输入模式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"dos_data 目录不存在: {data_dir}")

    outdir = args.outdir if args.outdir else data_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    index = scan_dos_data(data_dir)

    if args.interactive:
        (
            mode,
            element_kind,
            selected_elements,
            selected_sites,
            spd_together,
            group_subplots,
            xlim,
        ) = interactive_pick(index)
    else:
        mode = args.mode
        element_kind = args.element_kind
        spd_together = args.spd_together
        group_subplots = args.group_subplots
        xlim = tuple(args.xlim) if args.xlim else None

        available_elements = sorted(
            set(index.element_total.keys()) | set(index.element_spd.keys())
        )
        selected_elements = parse_elements(args.elements, available_elements)

        available_sites = sorted(index.site_spd.keys())
        selected_sites = parse_sites(args.sites, available_sites)

    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    all_outputs: list[Path] = []

    if mode == "element":
        if not selected_elements:
            raise ValueError("未选中任何有效元素，请检查 --elements")

        if element_kind in {"total", "both"}:
            total_panels = build_element_total_panels(index, selected_elements)
            all_outputs.extend(
                render_panels(
                    panels=total_panels,
                    outdir=outdir,
                    group_subplots=group_subplots,
                    xlim=xlim,
                    dpi=args.dpi,
                    show=args.show,
                    group_name="element_total",
                    cache=cache,
                )
            )

        if element_kind in {"spd", "both"}:
            spd_panels = build_element_spd_panels(index, selected_elements, spd_together)
            all_outputs.extend(
                render_panels(
                    panels=spd_panels,
                    outdir=outdir,
                    group_subplots=group_subplots,
                    xlim=xlim,
                    dpi=args.dpi,
                    show=args.show,
                    group_name="element_spd" if spd_together else "element_spd_split",
                    cache=cache,
                )
            )

    else:
        if not selected_sites:
            raise ValueError("未选中任何有效原子序号，请检查 --sites")

        site_spd_panels = build_site_spd_panels(index, selected_sites, spd_together)
        all_outputs.extend(
            render_panels(
                panels=site_spd_panels,
                outdir=outdir,
                group_subplots=group_subplots,
                xlim=xlim,
                dpi=args.dpi,
                show=args.show,
                group_name="site_spd" if spd_together else "site_spd_split",
                cache=cache,
            )
        )

    if not all_outputs:
        print("[Warning] 没有生成任何图片，请检查输入条件和数据文件。")
        return

    print(f"完成，共生成 {len(all_outputs)} 张图，输出目录: {outdir}")
    for p in all_outputs:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
