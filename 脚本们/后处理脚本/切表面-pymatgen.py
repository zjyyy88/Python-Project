"""
使用 pymatgen 切表面并批量导出 POSCAR 的脚本。

默认输入结构:
	D:\\aaazjy\\zjyyyyy\\halide water adsorption\\zjy-calc\\Bi-dopingLYC\\QJHCONTCAR-Bi0.33\\CONTCAR

主要功能:
1. 读取 CONTCAR/POSCAR 结构
2. 按给定 Miller 指数生成一个或多个终止面 (termination)
3. 可配置 slab 厚度、真空厚度、扩胞大小
4. 输出 VASP5 POSCAR，支持 Direct 或 Cartesian 坐标
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Poscar


# =============================
# 用户参数区（按需修改）
# =============================

INPUT_STRUCTURE = Path(
	r"D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\CONTCAR"
)

# 可放多个晶面指数，例如 [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
MILLER_LIST = [
	(1, 0, 0), (1, 1, 0), (1, 1, 1)
]

# 最小 slab 厚度（单位: Angstrom）
MIN_SLAB_SIZE = 15

# 最小真空厚度（单位: Angstrom）
MIN_VACUUM_SIZE = 15.0

# 是否居中 slab（True: 真空分布在两侧；False: 不强制居中）
CENTER_SLAB = True

# 其他几何参数
LLL_REDUCE = False
IN_UNIT_PLANES = False
PRIMITIVE = True
MAX_NORMAL_SEARCH = None
REORIENT_LATTICE = True

# 终止面处理
SYMMETRIZE = False
FILTER_OUT_SYM_SLABS = True

# 扩胞参数（必须包含 (1,1) 才会输出基础 1x1）
SUPERCELL_XY_LIST = [
	(1, 1),
	#(2, 1),
	(2, 2)
]

# 输出坐标模式，可选 "direct" 或 "cartesian"
WRITE_MODE = "direct"

# 输出目录
OUTPUT_DIR = Path("pymatgen_slab_outputs")


def sanitize_hkl(hkl: tuple[int, int, int]) -> str:
	return f"{hkl[0]}{hkl[1]}{hkl[2]}"


def write_poscar(structure: Structure, out_file: Path, mode: str) -> None:
	direct = mode.lower() == "direct"
	poscar_text = Poscar(structure, sort_structure=False).get_str(
		direct=direct,
		vasp4_compatible=False,
		significant_figures=16,
	)
	out_file.write_text(poscar_text, encoding="utf-8")


def valid_supercell_list(items: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
	result: list[tuple[int, int]] = []
	for sx, sy in items:
		if sx < 1 or sy < 1:
			raise ValueError(f"非法扩胞参数 {(sx, sy)}，要求 sx>=1 且 sy>=1")
		result.append((sx, sy))
	return result


def main() -> None:
	if not INPUT_STRUCTURE.exists():
		raise FileNotFoundError(f"输入结构不存在: {INPUT_STRUCTURE}")

	mode = WRITE_MODE.lower().strip()
	if mode not in {"direct", "cartesian"}:
		raise ValueError("WRITE_MODE 只支持 'direct' 或 'cartesian'")

	supercells = valid_supercell_list(SUPERCELL_XY_LIST)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	bulk = Structure.from_file(str(INPUT_STRUCTURE))
	a, b, c, alpha, beta, gamma = bulk.lattice.parameters
	print("=" * 80)
	print(f"输入结构: {INPUT_STRUCTURE}")
	print(f"化学式: {bulk.composition.reduced_formula}")
	print(
		"晶格参数: "
		f"a={a:.4f}, b={b:.4f}, c={c:.4f}, "
		f"alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}"
	)

	total_written = 0
	for hkl in MILLER_LIST:
		generator = SlabGenerator(
			initial_structure=bulk,
			miller_index=hkl,
			min_slab_size=MIN_SLAB_SIZE,
			min_vacuum_size=MIN_VACUUM_SIZE,
			lll_reduce=LLL_REDUCE,
			center_slab=CENTER_SLAB,
			in_unit_planes=IN_UNIT_PLANES,
			primitive=PRIMITIVE,
			max_normal_search=MAX_NORMAL_SEARCH,
			reorient_lattice=REORIENT_LATTICE,
		)

		slabs = generator.get_slabs(
			symmetrize=SYMMETRIZE,
			repair=False,
			filter_out_sym_slabs=FILTER_OUT_SYM_SLABS,
		)

		if not slabs:
			print(f"[hkl={hkl}] 未生成可用 slab，请调整参数。")
			continue

		print(f"[hkl={hkl}] 生成终止面数量: {len(slabs)}")

		for term_idx, slab in enumerate(slabs, start=1):
			for sx, sy in supercells:
				slab_out = slab.copy()
				if sx != 1 or sy != 1:
					slab_out.make_supercell([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

				fname = (
					f"POSCAR_hkl{sanitize_hkl(hkl)}_term{term_idx}_"
					f"SC{sx}x{sy}_{mode}.vasp"
				)
				out_file = OUTPUT_DIR / fname
				write_poscar(slab_out, out_file, mode)
				total_written += 1
				print(f"已保存: {out_file}")

	print("=" * 80)
	print(f"完成，总共输出 {total_written} 个 POSCAR 文件。")


if __name__ == "__main__":
	main()
