#!/usr/bin/env python3
"""
读取已汇总的电导率数据表（与 transport_summary.csv 同列名），做 Arrhenius 拟合，输出活化能和拟合图。
配置写在脚本顶部，直接运行即可。
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===== 配置区 ===== #
EXCEL_FILE = Path(r"E:/固态组/LiLa2O3/势函数/LiLaO2/电导率.xlsx")
SHEET_NAME = 0  # sheet 名称或索引
OUT_DIR = Path(r"E:/固态组/LiLa2O3/势函数/LiLaO2/arrhenius_sigma")
# σ 列单位，可选："S/cm", "mS/cm", "S/m"
SIGMA_UNIT = "mS/cm"
# 优先按列名选择电导率列（如 "sigma_S_per_cm" 或 "sigma_mS_per_cm"），否则可指定列索引（0-based）。
SIGMA_COL_NAME: str | None = "sigma_mS_per_cm"
SIGMA_COL_INDEX: int | None = None

K_B_EV = 8.617333262e-5  # eV/K
SIGMA_COL_CANDIDATES: Iterable[str] = (
    "sigma_S_per_cm",
    "sigma",
    "sigma (S/cm)",
    "sigma_S_cm",
    "sigma_S/cm",
)
TEMP_COL_CANDIDATES: Iterable[str] = ("temperature_K", "T", "temp")
TARGET_T_K = 300.0
# ================== #


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"找不到所需列，备选: {candidates}. 实际列: {list(df.columns)}")


def arrhenius_sigma_t(temperature_k: np.ndarray, sigma_s_cm: np.ndarray) -> dict:
    x = 1.0 / temperature_k
    y = np.log(sigma_s_cm * temperature_k)
    slope, intercept = np.polyfit(x, y, 1)
    ea_ev = -slope * K_B_EV
    return {
        "x": x,
        "y": y,
        "slope": slope,
        "intercept": intercept,
        "Ea_eV": ea_ev,
    }


def main() -> None:
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件: {EXCEL_FILE}")

    df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df = df_raw.copy()
    df = df.dropna(how="all")

    temp_col = pick_column(df, TEMP_COL_CANDIDATES)

    if SIGMA_COL_NAME is not None and SIGMA_COL_NAME in df.columns:
        sigma_series = df[SIGMA_COL_NAME]
        df = df[[temp_col]].copy()
        df["sigma_raw"] = sigma_series
    elif SIGMA_COL_INDEX is not None:
        if SIGMA_COL_INDEX < 0 or SIGMA_COL_INDEX >= df.shape[1]:
            raise IndexError(f"SIGMA_COL_INDEX 越界: {SIGMA_COL_INDEX}, 列数 {df.shape[1]}")
        sigma_series = df.iloc[:, SIGMA_COL_INDEX]
        df = df[[temp_col]].copy()
        df["sigma_raw"] = sigma_series
    else:
        sigma_col = pick_column(df, SIGMA_COL_CANDIDATES)
        df = df[[temp_col, sigma_col]].rename(columns={temp_col: "temperature_K", sigma_col: "sigma_raw"})
    df = df.dropna()

    if len(df) < 2:
        raise ValueError("温度点少于 2，无法拟合活化能")

    # 单位统一到 S/cm
    unit = SIGMA_UNIT.lower()
    if unit == "s/cm":
        factor = 1.0
    elif unit == "ms/cm":
        factor = 1e-3
    elif unit == "s/m":
        factor = 1.0 / 100.0  # S/m -> S/cm
    else:
        raise ValueError(f"SIGMA_UNIT 未识别: {SIGMA_UNIT}")

    df["sigma_S_per_cm"] = df["sigma_raw"].astype(float) * factor

    df = df.sort_values("temperature_K").reset_index(drop=True)
    fit = arrhenius_sigma_t(df["temperature_K"].to_numpy(dtype=float), df["sigma_S_per_cm"].to_numpy(dtype=float))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    txt = OUT_DIR / "activation_energy_from_sigmaT.txt"
    pred_ln_sigma_t = fit["intercept"] + fit["slope"] * (1.0 / TARGET_T_K)
    pred_sigma_s_cm = np.exp(pred_ln_sigma_t) / TARGET_T_K
    pred_sigma_ms_cm = pred_sigma_s_cm * 1e3

    with txt.open("w", encoding="utf-8") as f:
        f.write("Arrhenius 拟合 ln(σT) vs 1/T\n")
        f.write(f"Ea = {fit['Ea_eV']:.6f} eV\n")
        f.write(f"slope = {fit['slope']:.6e} (1/K)\n")
        f.write(f"intercept = {fit['intercept']:.6e}\n")
        f.write(f"sigma@{TARGET_T_K:.1f}K = {pred_sigma_s_cm:.6e} S/cm\n")
        f.write(f"sigma@{TARGET_T_K:.1f}K = {pred_sigma_ms_cm:.6e} mS/cm\n")

    x_line = np.linspace(fit["x"].min(), fit["x"].max(), 200)
    y_line = fit["slope"] * x_line + fit["intercept"]

    plt.figure(figsize=(6, 5))
    plt.scatter(fit["x"], fit["y"], label="data")
    for t, x_pt, y_pt in zip(df["temperature_K"], fit["x"], fit["y"]):
        plt.annotate(f"{t:.0f}K", xy=(x_pt, y_pt), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.plot(x_line, y_line, "r--", label=f"fit Ea={fit['Ea_eV']:.4f} eV")
    plt.xlabel("1/T (1/K)")
    plt.ylabel("ln(σT)")
    plt.title("Arrhenius fit from conductivity")
    plt.legend()
    plt.tight_layout()
    fig_path = OUT_DIR / "arrhenius_sigmaT.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("=== 拟合完成 ===")
    print(df.drop(columns=["sigma_raw"]).to_string(index=False))
    print(f"Ea = {fit['Ea_eV']:.6f} eV")
    print(f"sigma@{TARGET_T_K:.1f}K = {pred_sigma_s_cm:.6e} S/cm")
    print(f"sigma@{TARGET_T_K:.1f}K = {pred_sigma_ms_cm:.6e} mS/cm")
    print(f"结果: {txt}")
    print(f"图: {fig_path}")


if __name__ == "__main__":
    main()