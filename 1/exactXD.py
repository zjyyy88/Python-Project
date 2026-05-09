"""从 XDATCAR 批量提取多个步数并导出为 POSCAR_step_XXXX 文件。"""

from typing import List, Tuple, Set

xdatcar_file = "XDATCAR"

# 方式1：手动指定多个步数
#target_steps = [250, 500,750, 1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500]

# 方式2：用范围批量生成（取消注释即可）
target_steps = list(range(5000, 25000, 200))  # 1000,1100,...,2000


def parse_atom_count(count_line: str) -> int:
    return sum(int(item) for item in count_line.split())


def extract_steps_from_xdatcar(file_path: str, steps: List[int]) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 8:
        raise ValueError("XDATCAR 内容太短，格式可能不正确。")

    # XDATCAR 头部（POSCAR前7行：注释、缩放、3个晶格向量、元素名、元素数）
    header = lines[:7]
    atom_count = parse_atom_count(lines[6])

    # 找到所有构型起始行（Direct configuration= n）
    markers: List[Tuple[int, int]] = []
    for idx, line in enumerate(lines):
        if line.startswith("Direct configuration="):
            step_str = line.split("=")[-1].strip()
            try:
                step_no = int(step_str)
            except ValueError:
                continue
            markers.append((step_no, idx))

    if not markers:
        raise ValueError("未找到 'Direct configuration='，请检查 XDATCAR 格式。")

    wanted: Set[int] = set(steps)
    found: Set[int] = set()

    for step_no, start_idx in markers:
        if step_no not in wanted:
            continue

        coord_start = start_idx + 1
        coord_end = coord_start + atom_count
        if coord_end > len(lines):
            print(f"Step {step_no}: 原子坐标不完整，已跳过。")
            continue

        coords = lines[coord_start:coord_end]
        out_name = f"POSCAR_step_{step_no}"
        with open(out_name, "w", encoding="utf-8") as out:
            out.writelines(header)
            out.write("Direct\n")
            out.writelines(coords)

        found.add(step_no)
        print(f"已导出: {out_name}")

    missing = sorted(wanted - found)
    if missing:
        print(f"以下步数未找到: {missing}")


if __name__ == "__main__":
    extract_steps_from_xdatcar(xdatcar_file, target_steps)
