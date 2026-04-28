#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "按第7列x分组，在每组内选取第3列能量最小的整行，"
            "并输出CSV与PDEntry文本。"
        )
    )
    parser.add_argument(
        "--input",
        default=r"E:\固态组\LiLa2O3\La2O3-Lithiation\wz-SEND-convex\energy_per_formula_gcd(1).xlsx",
        help="输入文件路径（支持 .xlsx/.xls/.xlsm/.csv）",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel工作表名或索引（默认0，表示第一个工作表）",
    )
    parser.add_argument(
        "--x-col",
        type=int,
        default=7,
        help="用于分组的x列号（从1开始，默认7）",
    )
    parser.add_argument(
        "--formula-col",
        type=int,
        default=2,
        help="化学式列号（从1开始，默认2）",
    )
    parser.add_argument(
        "--energy-col",
        type=int,
        default=3,
        help="能量列号（从1开始，默认3）",
    )
    parser.add_argument(
        "--x-round",
        type=int,
        default=12,
        help="x分组前的小数保留位数，避免浮点微小误差（默认12）",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="输出CSV路径（默认与输入同目录，文件名加 _min_by_x.csv）",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="输出TXT路径（默认与输入同目录，文件名加 _pdentries.txt）",
    )
    return parser.parse_args()


def read_table(input_path: Path, sheet):
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(input_path, sheet_name=sheet)
    if suffix == ".csv":
        # 常见CSV编码回退顺序，兼容中文Windows环境
        for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(input_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("csv", b"", 0, 1, "无法使用常见编码读取CSV文件")
    raise ValueError("仅支持 .xlsx/.xls/.xlsm/.csv 文件")


def validate_columns(df: pd.DataFrame, *cols: int):
    if any(c < 1 for c in cols):
        raise ValueError("列号从1开始，所有列号必须 >= 1")

    max_col = max(cols)
    if df.shape[1] < max_col:
        raise ValueError(
            f"输入文件列数不足：至少需要第 {max_col} 列，实际仅 {df.shape[1]} 列"
        )


def select_min_rows_by_x(df: pd.DataFrame, x_col: int, energy_col: int, x_round: int):
    work = df.copy()
    work["_x"] = pd.to_numeric(work.iloc[:, x_col - 1], errors="coerce")
    work["_energy"] = pd.to_numeric(work.iloc[:, energy_col - 1], errors="coerce")

    work = work.dropna(subset=["_x", "_energy"]).copy()
    if work.empty:
        raise ValueError("可用数据为空：x列或能量列无法解析为数值")

    work["_x_group"] = work["_x"].round(x_round)
    idx = work.groupby("_x_group", sort=True)["_energy"].idxmin()
    selected = work.loc[idx].sort_values(by="_x", kind="mergesort").copy()
    return selected


def write_outputs(
    selected: pd.DataFrame,
    output_csv: Path,
    output_txt: Path,
    formula_col: int,
    energy_col: int,
):
    csv_df = selected.drop(columns=["_x", "_energy", "_x_group"], errors="ignore")
    csv_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    formulas = selected.iloc[:, formula_col - 1].astype(str).str.strip()
    energies = pd.to_numeric(selected.iloc[:, energy_col - 1], errors="coerce")

    lines = []
    for formula, energy in zip(formulas, energies):
        if pd.isna(energy):
            continue
        lines.append(f'PDEntry(Composition("{formula}"), {energy:.12g})')

    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_min_by_x.csv")
    )
    output_txt = (
        Path(args.output_txt)
        if args.output_txt
        else input_path.with_name(f"{input_path.stem}_pdentries.txt")
    )

    df = read_table(input_path, args.sheet)
    validate_columns(df, args.x_col, args.formula_col, args.energy_col)

    selected = select_min_rows_by_x(
        df=df,
        x_col=args.x_col,
        energy_col=args.energy_col,
        x_round=args.x_round,
    )

    write_outputs(
        selected=selected,
        output_csv=output_csv,
        output_txt=output_txt,
        formula_col=args.formula_col,
        energy_col=args.energy_col,
    )

    print(f"输入文件: {input_path}")
    print(f"共读取 {len(df)} 行，有效行 {len(selected)} 组x")
    print(f"最小能量整行CSV已输出: {output_csv}")
    print(f"PDEntry文本已输出: {output_txt}")


if __name__ == "__main__":
    main()
