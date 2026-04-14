import numpy as np
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path
import re
import ase.build
import ase.io


def cut_slab(bulk_structure, vector_a, vector_b, nlayers, vacuum=15, **kwargs):
    """
    根据给定的基矢量和层数切割晶体表面。

    参数:
    bulk_structure: 体相结构 (Atoms对象)
    vector_a: 定义表面的第一个基矢量 (在原晶包基矢坐标系下)
    vector_b: 定义表面的第二个基矢量
    nlayers: 切割的层数
    vacuum: 真空层总厚度 (A), 默认为15A
    **kwargs: 传递给 ase.build.cut 的其他参数 (如 origo)
    """
    slab_structure = ase.build.cut(
        bulk_structure,
        a=vector_a,
        b=vector_b,
        nlayers=nlayers,
        **kwargs
    )
    sorted_slab_structure = ase.build.sort(slab_structure)
    sorted_slab_structure.center(vacuum=vacuum / 2, axis=2)
    return sorted_slab_structure


def _format_counter(counter):
    """将元素计数器格式化为紧凑字符串，例如 Cl6Li3Y1。"""
    return "".join(f"{el}{counter[el]}" for el in sorted(counter))


def parse_formula_to_counter(formula):
    """把化学式字符串解析为元素计数器。"""
    parts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    counts = Counter()
    for element, number_text in parts:
        counts[element] += int(number_text) if number_text else 1
    return counts


def reduced_formula_key(formula):
    """返回化学式的最简计量键，例如 Bi2Cl36Li18Y4 -> ((Bi,1),(Cl,18),(Li,9),(Y,2))。"""
    counts = parse_formula_to_counter(formula)
    if not counts:
        return tuple()
    divisor = reduce(gcd, counts.values())
    return tuple((el, counts[el] // divisor) for el in sorted(counts))


def reduced_formula_key_to_text(key):
    """将最简计量键转换为可读字符串。"""
    return "".join(f"{el}{num}" for el, num in key)


def get_surface_termination_signature(slab_structure, surface_tol=1.2):
    """
    基于 z 方向上表面附近原子统计，生成顶部/底部终止面的签名。

    参数:
    slab_structure: Slab 结构对象
    surface_tol: 统计表面原子的厚度阈值 (A)
    """
    symbols = slab_structure.get_chemical_symbols()
    z_coords = slab_structure.positions[:, 2]
    z_min = float(np.min(z_coords))
    z_max = float(np.max(z_coords))

    bottom_counter = Counter(
        symbol for symbol, z in zip(symbols, z_coords) if z <= z_min + surface_tol
    )
    top_counter = Counter(
        symbol for symbol, z in zip(symbols, z_coords) if z >= z_max - surface_tol
    )

    return _format_counter(bottom_counter), _format_counter(top_counter)


def build_structure_signature(slab_structure, distance_decimals=3, surface_tol=1.2):
    """
    构建用于去重的结构签名。

    签名由以下部分组成：
    1) 化学式
    2) 晶胞长度与角度
    3) 顶/底表面终止面签名
    4) 周期性最短原子对距离列表 (按元素对+距离排序)

    这样可在“同化学式”条件下进一步区分不同表面终止与原子排布。
    """
    slab_cp = slab_structure.copy()
    slab_cp.wrap()

    formula = slab_cp.get_chemical_formula(mode="hill")
    lengths = tuple(round(float(v), 3) for v in slab_cp.cell.lengths())
    angles = tuple(round(float(v), 3) for v in slab_cp.cell.angles())
    bottom_sig, top_sig = get_surface_termination_signature(
        slab_cp, surface_tol=surface_tol
    )

    symbols = slab_cp.get_chemical_symbols()
    dist_matrix = slab_cp.get_all_distances(mic=True)

    pair_terms = []
    n_atoms = len(symbols)
    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            sym_i, sym_j = sorted((symbols[i], symbols[j]))
            dij = round(float(dist_matrix[i, j]), distance_decimals)
            pair_terms.append((sym_i, sym_j, dij))

    pair_terms.sort()
    return (formula, lengths, angles, bottom_sig, top_sig, tuple(pair_terms))


def _float_to_tag(value):
    """把浮点数转成适合文件名的标签。"""
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")


# ===================== 用户配置区 =====================
bulk_structure = ase.io.read(
    r'D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\Bi\CONTCAR_Bi3_CONTCAR'
)
outdir = Path(
    r'D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\Bi'
)
outdir.mkdir(parents=True, exist_ok=True)

vector = (1, 0, 0)
vector_a, vector_b = (0, 2, 0), (0, 0, 2)
nlayers = 27

origo_values = np.arange(-0.5, 0.5, 0.01)
target_formulas = {
    'BiCl18Li9Y2',
    'Bi2Cl36Li18Y4',
    'Bi3Cl54Li27Y6',
    'Bi4Cl72Li36Y8',
}
target_reduced_keys = {reduced_formula_key(formula) for formula in target_formulas}

surface_tol = 1.2
#含义：定义“表面层厚度阈值”，单位是 Å。
#作用：在统计上下表面终止面时，满足z坐标在 slab 的 z_min + surface_tol 以内的原子被认为是底部表面原子，满足 z 坐标在 slab 的 z_max - surface_tol 以外的原子被认为是顶部表面原子。
distance_decimals = 3
#含义：原子对距离保留的小数位数。
#作用：构建结构签名时会把距离四舍五入到 3 位小数，用于去重比较。
#小数位更多：判定更严格，更容易分成不同结构
#小数位更少：判定更宽松，更容易被合并为同一结构
# ======================================================


all_matches = []
observed_formula_counts = Counter()

for or_ in origo_values:
    slab_structure = cut_slab(
        bulk_structure,
        vector_a,
        vector_b,
        nlayers,
        origo=(or_, 0, or_ / 3),
    )
    formula = slab_structure.get_chemical_formula(mode='hill')
    observed_formula_counts[formula] += 1
    print(f"origo={or_:.3f}, Formula={formula}")

    formula_key = reduced_formula_key(formula)
    is_exact_match = formula in target_formulas
    is_reduced_match = formula_key in target_reduced_keys

    if is_exact_match or is_reduced_match:
        match_type = "exact" if is_exact_match else "reduced"
        bottom_sig, top_sig = get_surface_termination_signature(
            slab_structure, surface_tol=surface_tol
        )
        print(
            f"Match found ({match_type}): formula={formula}, origo={or_:.3f}, "
            f"bottom={bottom_sig}, top={top_sig}"
        )
        all_matches.append(
            {
                "origo": float(or_),
                "formula": formula,
                "formula_reduced": reduced_formula_key_to_text(formula_key),
                "match_type": match_type,
                "termination": (bottom_sig, top_sig),
                "slab": slab_structure,
            }
        )


if not all_matches:
    script_name = Path(__file__).stem
    result_dir = outdir / f"{script_name}_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)

    no_match_summary_path = result_dir / "no_match_summary.txt"
    with no_match_summary_path.open("w", encoding="utf-8") as f:
        f.write("=== No Match Summary ===\n")
        f.write(f"vector = {vector}\n")
        f.write(f"vector_a = {vector_a}, vector_b = {vector_b}\n")
        f.write(f"nlayers = {nlayers}\n")
        f.write(f"origo scan = [{origo_values[0]:.3f}, {origo_values[-1]:.3f}], step=0.01\n")
        f.write(f"target_formulas (exact) = {sorted(target_formulas)}\n")
        f.write(
            "target_formulas (reduced) = "
            f"{sorted(reduced_formula_key_to_text(k) for k in target_reduced_keys)}\n\n"
        )

        f.write("No slab matched target formulas in this scan range.\n\n")
        f.write("Top observed formulas:\n")
        for formula, count in observed_formula_counts.most_common(30):
            f.write(
                f"  {formula}: count={count}, "
                f"reduced={reduced_formula_key_to_text(reduced_formula_key(formula))}\n"
            )

    print("\n=== Done (No Match) ===")
    print("No slab matched target_formulas in the scanned origo range.")
    print(f"Diagnostic summary written to: {no_match_summary_path}")
    raise SystemExit(0)


unique_by_signature = {}
for match in all_matches:
    signature = build_structure_signature(
        match["slab"],
        distance_decimals=distance_decimals,
        surface_tol=surface_tol,
    )

    if signature not in unique_by_signature:
        unique_by_signature[signature] = {
            "formula": match["formula"],
            "formula_reduced": match["formula_reduced"],
            "match_type": match["match_type"],
            "termination": match["termination"],
            "origos": [match["origo"]],
            "slab": match["slab"],
        }
    else:
        unique_by_signature[signature]["origos"].append(match["origo"])


script_name = Path(__file__).stem
vector_tag = "_".join(map(str, vector))

result_dir = outdir / f"{script_name}_outputs"
all_match_dir = result_dir / "all_matches"
unique_dir = result_dir / "unique_surfaces"

all_match_dir.mkdir(parents=True, exist_ok=True)
unique_dir.mkdir(parents=True, exist_ok=True)


# 输出全部命中结构
for idx, match in enumerate(all_matches, start=1):
    file_name = (
        f"{idx:03d}_{vector_tag}_nlayers{nlayers}_{match['formula']}_"
        f"origo{_float_to_tag(match['origo'])}.vasp"
    )
    ase.io.write(all_match_dir / file_name, match["slab"], sort=True, direct=True)


# 输出去重后的唯一结构
unique_entries = list(unique_by_signature.values())
for idx, entry in enumerate(unique_entries, start=1):
    origos_sorted = sorted(entry["origos"])
    origo_min = origos_sorted[0]
    origo_max = origos_sorted[-1]
    file_name = (
        f"U{idx:02d}_{vector_tag}_nlayers{nlayers}_{entry['formula']}_"
        f"hits{len(origos_sorted)}_"
        f"origo{_float_to_tag(origo_min)}_to_{_float_to_tag(origo_max)}.vasp"
    )
    ase.io.write(unique_dir / file_name, entry["slab"], sort=True, direct=True)


summary_path = result_dir / "summary.txt"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("=== Slab Search Summary ===\n")
    f.write(f"vector = {vector}\n")
    f.write(f"vector_a = {vector_a}, vector_b = {vector_b}\n")
    f.write(f"nlayers = {nlayers}\n")
    f.write(f"origo scan = [{origo_values[0]:.3f}, {origo_values[-1]:.3f}], step=0.01\n")
    f.write(f"target_formulas = {sorted(target_formulas)}\n")
    f.write(
        "target_formulas_reduced = "
        f"{sorted(reduced_formula_key_to_text(k) for k in target_reduced_keys)}\n"
    )
    f.write(f"surface_tol = {surface_tol} A\n")
    f.write(f"distance_decimals = {distance_decimals}\n\n")

    f.write(f"Total matched slabs = {len(all_matches)}\n")
    f.write(f"Unique slabs after dedup = {len(unique_entries)}\n\n")

    for idx, entry in enumerate(unique_entries, start=1):
        origos_sorted = sorted(entry["origos"])
        origo_list_str = ", ".join(f"{v:.3f}" for v in origos_sorted)
        bottom_sig, top_sig = entry["termination"]

        f.write(f"[U{idx:02d}] formula={entry['formula']}\n")
        f.write(f"  formula_reduced = {entry['formula_reduced']}\n")
        f.write(f"  match_type = {entry['match_type']}\n")
        f.write(f"  termination(bottom/top) = {bottom_sig} / {top_sig}\n")
        f.write(f"  hit_count = {len(origos_sorted)}\n")
        f.write(f"  origos = {origo_list_str}\n\n")


print("\n=== Done ===")
print(f"Total matched slabs: {len(all_matches)}")
print(f"Unique slabs: {len(unique_entries)}")
print(f"All matched slabs dir: {all_match_dir}")
print(f"Unique slabs dir: {unique_dir}")
print(f"Summary file: {summary_path}")
