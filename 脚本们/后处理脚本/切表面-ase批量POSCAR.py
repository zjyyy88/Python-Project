"""
ASE 切表面批处理脚本（POSCAR/CONTCAR）

功能:
1. 自动识别输入目标（文件、目录，或“...\\Bi、CONTCAR_Bi3_CONTCAR”这种写法）
2. 基于 Miller 指数 + 层数生成 slab
3. 批量设置不同真空层厚度、不同 slab 层数
4. 自动扩胞（含 1x1 基础超胞）
5. 可选将倾斜表面胞近似转成更接近正交的胞，减少吸附位点标注和覆盖率统计复杂性
6. 自动将 slab 最低 z 对齐到 z=0，真空仅放在顶部（下实上虚）
7. 输出 VASP5 POSCAR，支持 Direct 或 Cartesian 坐标

运行前请确认:
- 已安装 ASE: pip install ase
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell, surface
from ase.io import read, write


# =============================
# 用户参数区（按需修改）
# =============================

# 你提供的输入路径风格（注意末尾用了中文顿号“、”）
INPUT_TARGETS: List[str] = [
    r"D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\Bi、CONTCAR_Bi3_CONTCAR",
]

# 晶面指数，可放多个
MILLER_LIST: List[Tuple[int, int, int]] = [
    (1, 0, 0),
]

# slab 层数列表
LAYER_LIST: List[int] = [4, 6]

# 顶部真空厚度列表（单位 Å）
VACUUM_LIST: List[float] = [10.0, 15.0]

# XY 扩胞列表，必须包含 (1, 1) 才有基础 1x1
SUPERCELL_XY_LIST: List[Tuple[int, int]] = [
    (1, 1),
    (2, 1),
    (2, 2),
]

# 输出坐标模式："direct" 或 "cartesian"；也可同时输出两种
WRITE_MODES: List[str] = ["direct"]

# 输出目录（相对路径会在当前工作目录下创建）
OUTPUT_DIR = Path("slab_poscar_outputs")

# 是否尝试将 in-plane 晶格变换为更接近正交（降低斜晶胞复杂性）
ORTHOGONALIZE_INPLANE = True
MAX_INPLANE_DET = 6
ANGLE_TOL_DEG = 2.0
MAX_COEFF = 3


# =============================
# 核心逻辑
# =============================

@dataclass
class SlabRecord:
    source_file: str
    formula: str
    elements: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    hkl: str
    layers: int
    vacuum_top: float
    supercell_xy: str
    coord_mode: str
    orthogonalized: bool
    output_file: str


def parse_special_input_target(target: str) -> List[Path]:
    """
    解析类似:
    D:\\...\\QJHCONTCAR-Bi0.33\\Bi、CONTCAR_Bi3_CONTCAR
    这种“同目录多个文件名”写法。
    """
    raw = Path(target)
    if raw.exists():
        return [raw]

    name = raw.name
    if any(sep in name for sep in ("、", "，", ",", ";", "；")):
        parent = raw.parent
        names = [x.strip() for x in re.split(r"[、，,;；]", name) if x.strip()]
        return [parent / n for n in names]

    return [raw]


def is_probably_poscar(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        read(str(path), format="vasp")
        return True
    except Exception:
        return False


def discover_poscar_files_in_dir(folder: Path) -> List[Path]:
    preferred = [
        "POSCAR",
        "CONTCAR",
        "CONTCAR_Bi3_CONTCAR",
        "Bi",
    ]
    found: List[Path] = []

    for name in preferred:
        p = folder / name
        if is_probably_poscar(p):
            found.append(p)

    for p in sorted(folder.iterdir()):
        if p in found or not p.is_file():
            continue
        if p.suffix == "" or p.name.upper().startswith(("POSCAR", "CONTCAR")):
            if is_probably_poscar(p):
                found.append(p)

    return found


def resolve_inputs(targets: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []

    for t in targets:
        candidates = parse_special_input_target(t)
        for c in candidates:
            if c.is_file() and is_probably_poscar(c):
                resolved.append(c)
            elif c.is_dir():
                resolved.extend(discover_poscar_files_in_dir(c))
            else:
                # 允许用户只给文件名，尝试在当前目录查找
                p = Path(c.name)
                if p.is_file() and is_probably_poscar(p):
                    resolved.append(p)

    # 去重并保持顺序
    uniq: List[Path] = []
    seen = set()
    for p in resolved:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def find_near_orthogonal_transform(
    slab: Atoms,
    max_coeff: int = 3,
    max_det: int = 6,
    angle_tol_deg: float = 2.0,
) -> Tuple[np.ndarray | None, float]:
    """
    在二维 in-plane 上搜索整数变换:
      [u]   [m n][a]
      [v] = [p q][b]
    目标是让 angle(u, v) 接近 90 度。
    """
    cell = slab.cell.array
    a = cell[0]
    b = cell[1]
    best_p = None
    best_err = float("inf")
    best_cost = float("inf")

    for m in range(-max_coeff, max_coeff + 1):
        for n in range(-max_coeff, max_coeff + 1):
            for p in range(-max_coeff, max_coeff + 1):
                for q in range(-max_coeff, max_coeff + 1):
                    det = m * q - n * p
                    if det <= 0 or det > max_det:
                        continue

                    u = m * a + n * b
                    v = p * a + q * b
                    if np.linalg.norm(u) < 1e-8 or np.linalg.norm(v) < 1e-8:
                        continue

                    ang = angle_deg(u, v)
                    err = abs(ang - 90.0)
                    cost = err + 0.02 * det + 0.001 * (np.linalg.norm(u) + np.linalg.norm(v))

                    if cost < best_cost:
                        best_cost = cost
                        best_err = err
                        best_p = np.array(
                            [
                                [m, n, 0],
                                [p, q, 0],
                                [0, 0, 1],
                            ],
                            dtype=int,
                        )

    if best_p is None:
        return None, float("inf")

    if best_err <= angle_tol_deg:
        return best_p, best_err
    return None, best_err


def maybe_orthogonalize_inplane(slab: Atoms) -> Tuple[Atoms, bool, str]:
    """尝试将 in-plane 变得更接近正交，不强制。"""
    cell = slab.cell.array
    init_angle = angle_deg(cell[0], cell[1])
    init_err = abs(init_angle - 90.0)

    if init_err <= ANGLE_TOL_DEG:
        return slab, False, f"in-plane 已接近正交 (gamma={init_angle:.2f}°)"

    pmat, best_err = find_near_orthogonal_transform(
        slab,
        max_coeff=MAX_COEFF,
        max_det=MAX_INPLANE_DET,
        angle_tol_deg=ANGLE_TOL_DEG,
    )
    if pmat is None:
        return slab, False, f"未找到满足阈值的正交化整数变换 (best_err={best_err:.2f}°)"

    new_slab = make_supercell(slab, pmat)
    new_angle = angle_deg(new_slab.cell.array[0], new_slab.cell.array[1])
    msg = f"正交化完成: gamma {init_angle:.2f}° -> {new_angle:.2f}°"
    return new_slab, True, msg


def align_bottom_and_put_vacuum_on_top(slab: Atoms, vacuum_top: float) -> Atoms:
    """
    将 slab 最低 z 面对齐到 z=0，真空全部放在顶部。
    """
    atoms = slab.copy()
    pos = atoms.get_positions()

    zmin = float(np.min(pos[:, 2]))
    pos[:, 2] -= zmin

    zmax = float(np.max(pos[:, 2]))
    slab_thickness = max(zmax, 1e-6)

    cell = atoms.cell.array.copy()
    cell[2] = np.array([0.0, 0.0, slab_thickness + vacuum_top], dtype=float)

    atoms.set_positions(pos)
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc((True, True, True))
    atoms.wrap(eps=1e-9)
    return atoms


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def write_poscar(atoms: Atoms, output_file: Path, mode: str) -> None:
    direct = mode.lower() == "direct"
    write(
        str(output_file),
        atoms,
        format="vasp",
        direct=direct,
        vasp5=True,
        sort=False,
    )


def build_single_input(
    input_file: Path,
    output_dir: Path,
    records: List[SlabRecord],
) -> None:
    bulk = read(str(input_file), format="vasp")

    formula = bulk.get_chemical_formula(mode="hill")
    elems = sorted(set(bulk.get_chemical_symbols()), key=lambda x: bulk.get_chemical_symbols().index(x))
    a, b, c, alpha, beta, gamma = bulk.cell.cellpar()

    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"材料化学式: {formula}")
    print(f"元素: {', '.join(elems)}")
    print(f"晶格参数: a={a:.4f}, b={b:.4f}, c={c:.4f}, alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")

    stem = sanitize_name(input_file.name)

    for hkl in MILLER_LIST:
        for layers in LAYER_LIST:
            slab0 = surface(bulk, hkl, layers, vacuum=0.0, periodic=True)
            orthogonalized = False
            ortho_note = ""

            if ORTHOGONALIZE_INPLANE:
                slab0, orthogonalized, ortho_note = maybe_orthogonalize_inplane(slab0)
                print(f"[{input_file.name}] hkl={hkl}, layers={layers}: {ortho_note}")

            for sx, sy in SUPERCELL_XY_LIST:
                if sx < 1 or sy < 1:
                    raise ValueError(f"非法扩胞参数: {(sx, sy)}，必须 >= 1")

                slab_xy = slab0.repeat((sx, sy, 1))

                for vacuum_top in VACUUM_LIST:
                    slab_final = align_bottom_and_put_vacuum_on_top(slab_xy, vacuum_top)

                    for mode in WRITE_MODES:
                        mode_norm = mode.lower()
                        if mode_norm not in {"direct", "cartesian"}:
                            raise ValueError(f"WRITE_MODES 仅支持 'direct' 或 'cartesian'，收到: {mode}")

                        hkl_tag = f"{hkl[0]}{hkl[1]}{hkl[2]}"
                        vac_tag = str(vacuum_top).replace(".", "p")
                        out_name = (
                            f"POSCAR_{stem}_hkl{hkl_tag}_L{layers}_V{vac_tag}_SC{sx}x{sy}_{mode_norm}.vasp"
                        )
                        out_path = output_dir / out_name

                        write_poscar(slab_final, out_path, mode_norm)
                        print(f"已保存: {out_path}")

                        records.append(
                            SlabRecord(
                                source_file=str(input_file),
                                formula=formula,
                                elements=",".join(elems),
                                a=float(a),
                                b=float(b),
                                c=float(c),
                                alpha=float(alpha),
                                beta=float(beta),
                                gamma=float(gamma),
                                hkl=str(hkl),
                                layers=layers,
                                vacuum_top=float(vacuum_top),
                                supercell_xy=f"{sx}x{sy}",
                                coord_mode=mode_norm,
                                orthogonalized=orthogonalized,
                                output_file=str(out_path),
                            )
                        )


def save_manifest(records: Sequence[SlabRecord], output_dir: Path) -> None:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source_file",
                "formula",
                "elements",
                "a",
                "b",
                "c",
                "alpha",
                "beta",
                "gamma",
                "hkl",
                "layers",
                "vacuum_top",
                "supercell_xy",
                "coord_mode",
                "orthogonalized",
                "output_file",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.source_file,
                    r.formula,
                    r.elements,
                    f"{r.a:.6f}",
                    f"{r.b:.6f}",
                    f"{r.c:.6f}",
                    f"{r.alpha:.6f}",
                    f"{r.beta:.6f}",
                    f"{r.gamma:.6f}",
                    r.hkl,
                    r.layers,
                    f"{r.vacuum_top:.3f}",
                    r.supercell_xy,
                    r.coord_mode,
                    str(r.orthogonalized),
                    r.output_file,
                ]
            )
    print(f"清单已保存: {manifest}")


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(INPUT_TARGETS)
    if not inputs:
        raise FileNotFoundError(
            "未找到可读取的 POSCAR/CONTCAR 输入文件。"
            "请检查 INPUT_TARGETS 路径是否正确，或将目标目录/文件改为可访问路径。"
        )

    print("将处理以下输入文件:")
    for p in inputs:
        print(f"  - {p}")

    records: List[SlabRecord] = []
    for f in inputs:
        build_single_input(f, output_dir, records)

    save_manifest(records, output_dir)
    print(f"完成，总计生成 {len(records)} 个 POSCAR 文件。")


if __name__ == "__main__":
    main()
