# 1. 导入所有依赖模块
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PDEntry, PDPlotter, PhaseDiagram
from pymatgen.core import Composition


# ------------------------- 用户配置区（开关版） -------------------------
# 可选: "compound" 或 "elemental"
# compound: 使用 CompoundPhaseDiagram，需要 TERMINAL_COMPOSITIONS
# elemental: 使用 PhaseDiagram，不需要 TERMINAL_COMPOSITIONS，但需要元素端点
PHASE_DIAGRAM_MODE = "compound"
SHOW_UNSTABLE = True

# 化合物条目（可批量添加）
# energy 为该 formula 对应的总能量（eV）
COMPOUND_ENTRY_DATA = [
    {"formula": "LiCl", "energy": -7.3857361, "name": "LiCl"},
    {"formula": "BiCl3", "energy": -13.8687192, "name": "BiCl3"},
    {"formula": "YCl3", "energy": -21.17910362, "name": "YCl3"},
    {"formula": "Li27Y8Bi1Cl54", "energy": -380.96396106, "name": "target_1"},
    {"formula": "Li27Y7Bi2Cl54", "energy": -373.1618582, "name": "target_4"},
     {"formula": "Li9Y2Cl18Bi1", "energy":-121.7892274, "name": "target_2"},
    {"formula": "Li9Y3Cl18", "energy": -129.59261637, "name": "target_3"},
]

# compound 模式端点（可按需要改）
TERMINAL_COMPOSITIONS = ["LiCl", "BiCl3", "YCl3"]

# elemental 模式元素端点（每原子参考能）
ELEMENTAL_REFERENCE_DATA = [
    {"formula": "Li", "energy": -1.92, "name": "Li_ref"},
    {"formula": "Bi", "energy": -3.897912597, "name": "Bi_ref"},
    {"formula": "Y", "energy": -6.455346885, "name": "Y_ref"},
    {"formula": "Cl", "energy": -2.282874801, "name": "Cl_ref"},
]

# 多目标相评估配置
# 1) TARGET_FORMULAS 不为空: 只评估这里列出的目标相
# 2) TARGET_FORMULAS 为空且 AUTO_EVALUATE_ALL_COMPOUNDS=True:
#    自动评估所有“非端点”化合物
TARGET_FORMULAS = [
    "Li27Y8Bi1Cl54",
     "Li9Y2Cl18Bi1",
     "Li9Y3Cl18",
     "Li27Y7Bi2Cl54",
]
AUTO_EVALUATE_ALL_COMPOUNDS = False

# 可选: 导出多目标凸包能汇总
# 例: HULL_RESULTS_CSV = "multi_target_hull_results.csv"
HULL_RESULTS_CSV = None
# ----------------------------------------------------------------------


def build_entries_from_data(entry_data: list[dict]) -> list[PDEntry]:
    entries: list[PDEntry] = []
    for row in entry_data:
        entries.append(PDEntry(Composition(row["formula"]), float(row["energy"]), name=row.get("name")))
    return entries


def build_phase_diagram(mode: str, compound_entries: list[PDEntry], elemental_entries: list[PDEntry]):
    """按模式构建 PhaseDiagram 或 CompoundPhaseDiagram。"""
    mode = mode.lower().strip()

    if mode == "compound":
        terminal_comps = [Composition(formula) for formula in TERMINAL_COMPOSITIONS]
        pd = CompoundPhaseDiagram(compound_entries, terminal_comps)
        return pd, compound_entries

    if mode == "elemental":
        entries = elemental_entries + compound_entries
        pd = PhaseDiagram(entries)
        return pd, entries

    raise ValueError('PHASE_DIAGRAM_MODE 只能是 "compound" 或 "elemental"')


def find_target_entry(pd, entries: list[PDEntry], target_comp: Composition, mode: str):
    """在不同模式中定位目标相条目。"""
    rf = target_comp.reduced_formula
    mode = mode.lower().strip()

    if mode == "compound":
        candidates = [
            e
            for e in pd.all_entries
            if hasattr(e, "original_entry") and e.original_entry.composition.reduced_formula == rf
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x.original_entry.energy_per_atom)

    candidates = [e for e in entries if e.composition.reduced_formula == rf]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x.energy_per_atom)


def collect_target_formulas(compound_entries: list[PDEntry]) -> list[str]:
    if TARGET_FORMULAS:
        return TARGET_FORMULAS

    if not AUTO_EVALUATE_ALL_COMPOUNDS:
        return []

    terminal_set = {Composition(x).reduced_formula for x in TERMINAL_COMPOSITIONS}
    formulas: list[str] = []
    seen: set[str] = set()
    for entry in compound_entries:
        rf = entry.composition.reduced_formula
        if rf in terminal_set:
            continue
        if rf not in seen:
            formulas.append(rf)
            seen.add(rf)
    return formulas


def write_hull_results(rows: list[dict], output_csv: str) -> None:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["target_formula", "status", "e_above_hull_eV_per_atom", "is_stable"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"已导出多目标凸包能结果: {output.resolve()}")


def main():
    compound_entries = build_entries_from_data(COMPOUND_ENTRY_DATA)
    elemental_entries = build_entries_from_data(ELEMENTAL_REFERENCE_DATA)

    pd, entries_used = build_phase_diagram(PHASE_DIAGRAM_MODE, compound_entries, elemental_entries)
    print(f"运行模式: {PHASE_DIAGRAM_MODE}")
    print(f"总 entries 数量: {len(entries_used)}")

    # 1) 稳定相
    print("✅ 凸包上的稳定相：")
    for entry in pd.stable_entries:
        shown_entry = entry.original_entry if hasattr(entry, "original_entry") else entry
        print(f"成分：{shown_entry.composition}，单原子能量：{shown_entry.energy_per_atom:.2f} eV/atom")

    # 2) 多目标相凸包能
    target_formulas = collect_target_formulas(compound_entries)
    print("=" * 60)
    if not target_formulas:
        print("⚠ 未配置目标相。请填写 TARGET_FORMULAS，或开启 AUTO_EVALUATE_ALL_COMPOUNDS。")
    else:
        print(f"将评估目标相数量: {len(target_formulas)}")

    result_rows: list[dict] = []
    for formula in target_formulas:
        target_comp = Composition(formula)
        target_entry = find_target_entry(pd, entries_used, target_comp, PHASE_DIAGRAM_MODE)

        if target_entry is None:
            if PHASE_DIAGRAM_MODE.lower().strip() == "compound":
                print(f"⚠ {target_comp}: 不在当前 terminal_compositions 张成的相图空间内")
                status = "not_in_compound_space"
            else:
                print(f"⚠ {target_comp}: 在当前 entries 中未找到")
                status = "not_found_in_entries"

            result_rows.append(
                {
                    "target_formula": target_comp.reduced_formula,
                    "status": status,
                    "e_above_hull_eV_per_atom": "",
                    "is_stable": "",
                }
            )
            continue

        delta_e_hull = pd.get_e_above_hull(target_entry)
        is_stable = abs(delta_e_hull) < 1e-12
        print(
            f"✅ {target_comp.reduced_formula}: ΔE_hull = {delta_e_hull:.4f} eV/atom | "
            f"{'稳定' if is_stable else '亚稳'}"
        )
        result_rows.append(
            {
                "target_formula": target_comp.reduced_formula,
                "status": "ok",
                "e_above_hull_eV_per_atom": f"{delta_e_hull:.8f}",
                "is_stable": str(is_stable),
            }
        )

    if HULL_RESULTS_CSV and result_rows:
        write_hull_results(result_rows, HULL_RESULTS_CSV)

    # 3) 绘图
    plotter = PDPlotter(pd, show_unstable=SHOW_UNSTABLE)
    plotter.show()
    plt.title(f"体系 0K 凸包相图 ({PHASE_DIAGRAM_MODE})", fontsize=12)
    plt.xlabel("成分坐标", fontsize=10)
    plt.ylabel("形成能 (eV/atom)", fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()