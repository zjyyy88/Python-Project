#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于不同 Li 含量下的总能，按 ΔE(x)=E(x)-xE(1)-(1-x)E(0) 绘制组分图与下凸包。"
    )
    parser.add_argument(
        "--input",
        default=r"E:\固态组\LiLa2O3\La2O3-Lithiation\wz-SEND-convex\energy_per_formula_gcd(1).xlsx",
        help="输入数据文件（CSV 或 Excel）。默认读取桌面 energy_per_formula_gcd.excel",
    )
    parser.add_argument(
        "--e-x0",
        type=float,
        default=-1536.803308,
        help="x=0 的参考总能 E(x=0)；默认 -671.0190748",
    )
    parser.add_argument(
        "--e-x1",
        type=float,
        default=-1566.481201,
        help="x=1 的参考总能 E(x=1)；若不提供则自动从输入数据中读取",
    )
    parser.add_argument(
        "--output",
        default="PD_LixLa2O3.png",
        help="输出图像文件名（默认: PD_LixLaO3.png）",
    )
    parser.add_argument(
        "--x-col",
        type=int,
        default=7,
        help="组分 x 所在列（从 1 开始计数，默认第2列）",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=1,
        help="标签所在列（从 1 开始计数，默认第1列）",
    )
    parser.add_argument(
        "--energy-col",
        type=int,
        default=3,
        help="能量所在列（从 1 开始计数，默认第3列）",
    )
    parser.add_argument(
        "--header",
        choices=["auto", "none"],
        default="auto",
        help="是否把首行当表头：auto=自动(默认)，none=无表头",
    )
    return parser.parse_args()


def _read_from_csv(path, label_col, x_col, energy_col, has_header):
    points = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        if has_header:
            reader = csv.reader(f)
            next(reader, None)
        else:
            reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < max(label_col, x_col, energy_col):
                continue
            try:
                label = str(row[label_col - 1]).strip()
                x = float(row[x_col - 1])
                energy = float(row[energy_col - 1])
            except ValueError:
                continue
            points.append((label, x, energy))

    return points


def _read_from_excel(path, label_col, x_col, energy_col, has_header):
    header = 0 if has_header else None
    df = pd.read_excel(path, header=header)
    if df.shape[1] < max(label_col, x_col, energy_col):
        raise ValueError(
            f"文件列数不足：至少需要到第 {max(label_col, x_col, energy_col)} 列，实际仅 {df.shape[1]} 列"
        )

    label_series = df.iloc[:, label_col - 1].astype(str).str.strip()
    x_series = pd.to_numeric(df.iloc[:, x_col - 1], errors="coerce")
    e_series = pd.to_numeric(df.iloc[:, energy_col - 1], errors="coerce")
    valid = ~(x_series.isna() | e_series.isna())
    return list(zip(label_series[valid].tolist(), x_series[valid].tolist(), e_series[valid].tolist()))


def read_data(path, label_col=1, x_col=2, energy_col=3, header_mode="auto"):
    if label_col < 1 or x_col < 1 or energy_col < 1:
        raise ValueError("列号从 1 开始，--label-col、--x-col 和 --energy-col 必须 >= 1")

    has_header = header_mode == "auto"
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm", ".excel"}:
        points = _read_from_excel(path, label_col, x_col, energy_col, has_header)
    elif suffix == ".csv":
        points = _read_from_csv(path, label_col, x_col, energy_col, has_header)
    else:
        raise ValueError("仅支持 CSV 或 Excel 文件（.csv/.xlsx/.xls/.xlsm/.excel）")

    if not points:
        raise ValueError("输入文件中没有可用数值数据")
    return points


def deduplicate_by_x_min_energy(points):
    best = {}
    for label, x, e in points:
        if x not in best or e < best[x][2]:
            best[x] = (label, x, e)
    return sorted(best.values(), key=lambda t: t[1])


def find_e_x0(points, e_x0_from_arg=None, tol=1e-8):
    if e_x0_from_arg is not None:
        return e_x0_from_arg

    for _label, x, e in points:
        if abs(x) < tol:
            return e
    raise ValueError("未提供 --e-x0，且输入文件中未找到 x=0 的参考能量")


def find_e_x1(points, e_x1_from_arg=None, tol=1e-8):
    if e_x1_from_arg is not None:
        return e_x1_from_arg

    for _label, x, e in points:
        if abs(x - 1.0) < tol:
            return e
    raise ValueError("未提供 --e-x1，且输入文件中未找到 x=1 的参考能量")


def calc_formation_energies(points, e_x0, e_x1):
    # ΔE(x) = E(x) - x * E(x=1) - (1-x) * E(x=0)
    return [(label, x, e - x * e_x1 - (1 - x) * e_x0) for label, x, e in points]


def normalize_endpoints_to_zero(points_xy):
    if len(points_xy) < 2:
        return points_xy

    sorted_points = sorted(points_xy, key=lambda p: p[1])
    _label_left, x_left, y_left = sorted_points[0]
    _label_right, x_right, y_right = sorted_points[-1]

    if abs(x_right - x_left) < 1e-12:
        return points_xy

    def baseline(x):
        return y_left + (y_right - y_left) * (x - x_left) / (x_right - x_left)

    return [(label, x, y - baseline(x)) for label, x, y in points_xy]


def cross(o, a, b):
    return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])


def lower_hull(points_xy):
    # 输入需按 x 递增
    hull = []
    for p in sorted(points_xy, key=lambda t: t[1]):
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    raw_points = read_data(
        input_path,
        label_col=args.label_col,
        x_col=args.x_col,
        energy_col=args.energy_col,
        header_mode=args.header,
    )
    unique_points = deduplicate_by_x_min_energy(raw_points)

    e_x0 = find_e_x0(unique_points, args.e_x0)
    e_x1 = find_e_x1(unique_points, args.e_x1)

    form_points_all = calc_formation_energies(raw_points, e_x0, e_x1)
    form_points_unique = calc_formation_energies(unique_points, e_x0, e_x1)

    hull = lower_hull(form_points_unique)

    xs = [p[1] for p in form_points_all]
    ys = [p[2] for p in form_points_all]
    hx = [p[1] for p in hull]
    hy = [p[2] for p in hull]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=35, alpha=0.75, label="All configurations")
    plt.plot(hx, hy, "r-", lw=2.2, label="Lower convex hull (stable)")
    plt.xlabel("Li content x in LixLaO3")
    plt.ylabel("ΔE(x) = E(x)-xE(1)-(1-x)E(0)/(eV/f.u.)")
    plt.title("LixLa2O3: convex hull with tie-line energy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)

    print(f"已保存图像: {args.output}")
    print("\n稳定组分点（下凸包）:")
    for label, x, y in hull:
        print(f"label={label}, x={x:.6g}, ΔE={y:.6f}")


if __name__ == "__main__":
    main()
