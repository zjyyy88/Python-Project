"""
使用 pymatgen 计算相图/形成能/凸包能 的模块化脚本。

核心特点：
1. 支持从脚本内置数据、JSON 或 CSV 读取相条目。
2. 支持模块化运行：stable / formation / hull / plot。
3. 自动处理 total 与 per_atom 两种能量输入格式。
4. 内置详细命令教程：python 相图绝对能量-标明原子.py --tutorial

快速开始：
1) 使用脚本内置数据（默认模块: stable formation hull）
   python 相图绝对能量-标明原子.py

2) 只做凸包能计算（不输出其他模块）
   python 相图绝对能量-标明原子.py --modules hull

3) 计算并显示相图
   python 相图绝对能量-标明原子.py --modules plot --show-plot --show-unstable

4) 指定目标相并计算该相凸包能
   python 相图绝对能量-标明原子.py --modules hull --target-formula Li24Y8Bi1Cl54

5) 从 JSON 读入条目
   python 相图绝对能量-标明原子.py --entries-file entries.json --modules stable formation hull

6) 从 CSV 读入条目并导出结果表
   python 相图绝对能量-标明原子.py --entries-file entries.csv --formation-csv formation.csv --hull-csv hull.csv

JSON 条目格式示例：
[
  {"formula": "Li", "energy": -1.92, "energy_type": "per_atom", "name": "Li_metal"},
  {"formula": "Cl2", "energy": -4.60, "energy_type": "total", "n_atoms": 2, "name": "Cl2_gas"},
  {"formula": "LiCl", "energy": -7.38, "energy_type": "total", "n_atoms": 2, "name": "LiCl"}
]

CSV 条目格式示例（表头必须包含 formula,energy）：
formula,energy,energy_type,n_atoms,name
Li,-1.92,per_atom,,Li_metal
Cl2,-4.60,total,2,Cl2_gas
LiCl,-7.38,total,2,LiCl
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pymatgen.analysis.phase_diagram import PDEntry, PDPlotter, PhaseDiagram
from pymatgen.core import Composition


MODULE_CHOICES = ("stable", "formation", "hull", "plot")

TUTORIAL_TEXT = """
======================== 使用教程 ========================
一、能量输入规则
1. energy_type=total:  energy 是该条目对应化学式/晶胞总能量 (eV)
2. energy_type=per_atom: energy 是每原子能量 (eV/atom)
3. n_atoms 仅在 energy_type=total 时有意义。
   - 若 n_atoms 与 formula 原子数不同，脚本会按每原子能量换算到 formula。

二、模块说明（--modules）
1. stable: 输出凸包稳定相
2. formation: 输出所有条目的形成能（eV/formula 与 eV/atom）
3. hull: 输出所有条目的 E_above_hull；可额外计算目标相
4. plot: 绘制相图（可显示/保存）

三、常用命令
1) 查看教程
   python 相图绝对能量-标明原子.py --tutorial

2) 默认运行（stable + formation + hull）
   python 相图绝对能量-标明原子.py

3) 仅计算凸包能
   python 相图绝对能量-标明原子.py --modules hull

4) 仅稳定相 + 形成能
   python 相图绝对能量-标明原子.py --modules stable formation

5) 绘图并显示亚稳相
   python 相图绝对能量-标明原子.py --modules plot --show-plot --show-unstable

6) 绘图并保存文件
   python 相图绝对能量-标明原子.py --modules plot --save-plot pd.png --show-unstable

7) 指定目标相，仅根据条目库中的同化学计量最低能条目算凸包能
   python 相图绝对能量-标明原子.py --modules hull --target-formula Li24Y8Bi1Cl54

8) 指定目标相与目标能量（不必在条目库中）
   python 相图绝对能量-标明原子.py --modules hull --target-formula Li24Y8Bi1Cl54 --target-energy -380.96396106 --target-energy-type total --target-n-atoms 87

9) 从 JSON/CSV 读取条目
   python 相图绝对能量-标明原子.py --entries-file entries.json --modules stable formation hull
   python 相图绝对能量-标明原子.py --entries-file entries.csv --modules hull

10) 导出结果表
   python 相图绝对能量-标明原子.py --formation-csv formation.csv --hull-csv hull.csv
=========================================================
""".strip()


DEFAULT_ENTRY_DATA = [
    # 元素参考（建议使用同一计算设置得到的稳定单质参考）
    {"formula": "Li", "energy": -1.92, "energy_type": "per_atom", "name": "Li_ref"},
    {"formula": "Bi", "energy": -3.897912597, "energy_type": "per_atom", "name": "Bi_ref"},
    {"formula": "Y", "energy": -6.455346885, "energy_type": "per_atom", "name": "Y_ref"},
    {"formula": "Cl", "energy": -2.282874801, "energy_type": "per_atom", "name": "Cl_ref"},
    # 体系中间相
    {"formula": "LiCl", "energy": -7.3857361, "energy_type": "total", "n_atoms": 2, "name": "LiCl"},
    {"formula": "BiCl3", "energy": -14.32141014, "energy_type": "total", "n_atoms": 4, "name": "BiCl3"},
    {"formula": "YCl3", "energy": -21.17910362, "energy_type": "total", "n_atoms": 4, "name": "YCl3"},
    # 目标相（按 formula 计原子数应为 87）
    {
        "formula": "Li24Y8Bi1Cl54",
        "energy": -380.96396106,
        "energy_type": "total",
        "n_atoms": 87,
        "name": "target_phase",
    },
]


@dataclass
class EntrySpec:
    formula: str
    energy: float
    energy_type: str = "total"
    n_atoms: float | None = None
    name: str | None = None

    def to_pd_entry(self) -> PDEntry:
        """Convert EntrySpec to PDEntry; PDEntry energy must be total energy of this formula."""
        comp = Composition(self.formula)
        formula_atoms = float(comp.num_atoms)

        if self.energy_type not in {"total", "per_atom"}:
            raise ValueError(f"Unsupported energy_type={self.energy_type!r}. Use total or per_atom.")

        if self.energy_type == "per_atom":
            total_energy_for_formula = float(self.energy) * formula_atoms
            return PDEntry(comp, total_energy_for_formula, name=self.name)

        if self.n_atoms is None:
            source_atoms = formula_atoms
        else:
            source_atoms = float(self.n_atoms)
            if source_atoms <= 0:
                raise ValueError(f"n_atoms must be > 0 for formula {self.formula}.")

        per_atom_energy = float(self.energy) / source_atoms
        total_energy_for_formula = per_atom_energy * formula_atoms

        if self.n_atoms is not None and abs(source_atoms - formula_atoms) > 1e-8:
            print(
                f"[Warning] {self.formula}: n_atoms={source_atoms:g} 与 formula 原子数={formula_atoms:g} 不一致；"
                f"已按每原子能量换算到该 formula。"
            )

        return PDEntry(comp, total_energy_for_formula, name=self.name)


def load_entry_specs(entries_file: Path | None) -> list[EntrySpec]:
    if entries_file is None:
        return [EntrySpec(**item) for item in DEFAULT_ENTRY_DATA]

    if not entries_file.exists():
        raise FileNotFoundError(f"Entries file not found: {entries_file}")

    suffix = entries_file.suffix.lower()
    if suffix == ".json":
        with entries_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("JSON must be a list of entry objects.")
        return [EntrySpec(**item) for item in raw]

    if suffix == ".csv":
        specs: list[EntrySpec] = []
        with entries_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"formula", "energy"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")

            for row in reader:
                formula = (row.get("formula") or "").strip()
                if not formula:
                    continue
                energy = float(row["energy"])
                energy_type = (row.get("energy_type") or "total").strip() or "total"
                n_atoms_raw = (row.get("n_atoms") or "").strip()
                name = (row.get("name") or "").strip() or None
                n_atoms = float(n_atoms_raw) if n_atoms_raw else None
                specs.append(
                    EntrySpec(
                        formula=formula,
                        energy=energy,
                        energy_type=energy_type,
                        n_atoms=n_atoms,
                        name=name,
                    )
                )
        return specs

    raise ValueError("Only .json or .csv entries file is supported.")


def normalize_modules(raw_modules: list[str]) -> list[str]:
    modules = [m.lower() for m in raw_modules]
    if "all" in modules:
        return list(MODULE_CHOICES)
    for m in modules:
        if m not in MODULE_CHOICES:
            raise ValueError(f"Unknown module: {m}. Choices: {MODULE_CHOICES} or all")
    # 去重并保留输入顺序
    unique: list[str] = []
    seen = set()
    for m in modules:
        if m not in seen:
            unique.append(m)
            seen.add(m)
    return unique


def find_lowest_entry_for_formula(entries: list[PDEntry], formula: str) -> PDEntry:
    target = Composition(formula).reduced_composition
    matches = [e for e in entries if e.composition.reduced_composition == target]
    if not matches:
        raise ValueError(f"No entry found for target formula: {formula}")
    return min(matches, key=lambda e: e.energy_per_atom)


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 pymatgen 计算相图、形成能和凸包能（支持模块化选择）。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--entries-file", type=Path, default=None, help="条目输入文件（.json 或 .csv）")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["stable", "formation", "hull"],
        help="选择模块: stable formation hull plot 或 all",
    )

    parser.add_argument("--target-formula", type=str, default=None, help="目标相化学式（用于单独凸包能评估）")
    parser.add_argument("--target-energy", type=float, default=None, help="目标相能量")
    parser.add_argument(
        "--target-energy-type",
        choices=["total", "per_atom"],
        default="total",
        help="目标相能量类型（仅当 --target-energy 给出时生效）",
    )
    parser.add_argument("--target-n-atoms", type=float, default=None, help="目标相能量对应原子数（total 模式可选）")

    parser.add_argument("--show-unstable", action="store_true", help="绘图时显示亚稳相")
    parser.add_argument("--show-plot", action="store_true", help="绘图后弹窗显示")
    parser.add_argument("--save-plot", type=Path, default=None, help="保存相图路径，如 pd.png")

    parser.add_argument("--formation-csv", type=Path, default=None, help="导出形成能表 CSV")
    parser.add_argument("--hull-csv", type=Path, default=None, help="导出凸包能表 CSV")
    parser.add_argument("--tutorial", action="store_true", help="打印详细教程并退出")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.tutorial:
        print(TUTORIAL_TEXT)
        return

    modules = normalize_modules(args.modules)
    specs = load_entry_specs(args.entries_file)
    if not specs:
        raise ValueError("No entries provided.")

    pd_entries = [spec.to_pd_entry() for spec in specs]
    phase_diagram = PhaseDiagram(pd_entries)

    print("=" * 78)
    print("Phase diagram build finished")
    print(f"Total entries: {len(pd_entries)}")
    print(f"Chemical system: {phase_diagram.chemical_system}")
    print("=" * 78)

    if "stable" in modules:
        print("\n[Module: stable] 凸包稳定相")
        for entry in sorted(phase_diagram.stable_entries, key=lambda x: x.composition.reduced_formula):
            print(
                f"{entry.composition.formula:>24s} | "
                f"E_total={entry.energy:12.6f} eV | "
                f"E_atom={entry.energy_per_atom:10.6f} eV/atom"
            )

    formation_rows: list[dict] = []
    if "formation" in modules or args.formation_csv is not None:
        print("\n[Module: formation] 形成能")
        for entry in sorted(pd_entries, key=lambda x: x.composition.reduced_formula):
            row = {
                "formula": entry.composition.formula,
                "reduced_formula": entry.composition.reduced_formula,
                "energy_total_eV": entry.energy,
                "energy_per_atom_eV": entry.energy_per_atom,
                "formation_energy_eV_per_formula": phase_diagram.get_form_energy(entry),
                "formation_energy_eV_per_atom": phase_diagram.get_form_energy_per_atom(entry),
            }
            formation_rows.append(row)
            if "formation" in modules:
                print(
                    f"{row['formula']:>24s} | "
                    f"dE_form={row['formation_energy_eV_per_atom']:10.6f} eV/atom | "
                    f"dE_form_formula={row['formation_energy_eV_per_formula']:12.6f} eV"
                )

        if args.formation_csv is not None:
            write_csv(formation_rows, args.formation_csv)
            print(f"形成能表已导出: {args.formation_csv.resolve()}")

    hull_rows: list[dict] = []
    if "hull" in modules or args.hull_csv is not None:
        print("\n[Module: hull] 凸包能")
        for entry in sorted(pd_entries, key=lambda x: x.composition.reduced_formula):
            e_hull = phase_diagram.get_e_above_hull(entry)
            row = {
                "formula": entry.composition.formula,
                "reduced_formula": entry.composition.reduced_formula,
                "energy_total_eV": entry.energy,
                "energy_per_atom_eV": entry.energy_per_atom,
                "e_above_hull_eV_per_atom": e_hull,
                "is_stable": abs(e_hull) < 1e-12,
            }
            hull_rows.append(row)
            if "hull" in modules:
                print(
                    f"{row['formula']:>24s} | "
                    f"E_hull={row['e_above_hull_eV_per_atom']:10.6f} eV/atom | "
                    f"stable={row['is_stable']}"
                )

        if args.target_formula:
            if args.target_energy is None:
                target_entry = find_lowest_entry_for_formula(pd_entries, args.target_formula)
                source_msg = "(来自条目库同化学计量最低能条目)"
            else:
                target_spec = EntrySpec(
                    formula=args.target_formula,
                    energy=args.target_energy,
                    energy_type=args.target_energy_type,
                    n_atoms=args.target_n_atoms,
                    name="target_custom",
                )
                target_entry = target_spec.to_pd_entry()
                source_msg = "(来自命令行输入目标能量)"

            target_e_hull = phase_diagram.get_e_above_hull(target_entry)
            target_form = phase_diagram.get_form_energy_per_atom(target_entry)
            print("\n目标相评估")
            print(f"formula: {args.target_formula} {source_msg}")
            print(f"E_above_hull: {target_e_hull:.6f} eV/atom")
            print(f"Formation energy: {target_form:.6f} eV/atom")
            print("Stability:", "on hull (stable)" if abs(target_e_hull) < 1e-12 else "above hull (metastable)")

        if args.hull_csv is not None:
            write_csv(hull_rows, args.hull_csv)
            print(f"凸包能表已导出: {args.hull_csv.resolve()}")

    if "plot" in modules or args.save_plot is not None or args.show_plot:
        print("\n[Module: plot] 绘制相图")
        plotter = PDPlotter(phase_diagram, show_unstable=args.show_unstable, backend="matplotlib")
        plot_obj = plotter.get_plot()

        if hasattr(plot_obj, "figure"):
            fig = plot_obj.figure
        else:
            fig = plot_obj

        if args.save_plot is not None:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_plot, dpi=300, bbox_inches="tight")
            print(f"相图已保存: {args.save_plot.resolve()}")

        if args.show_plot:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()