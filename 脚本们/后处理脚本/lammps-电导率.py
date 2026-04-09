#!/usr/bin/env python3
"""
用已有的 LAMMPS MSD 表（两列：TimeStep，MSD(Å^2)）直接计算扩散系数和电导率。
配置写在脚本顶部，直接运行即可。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure

# ==== 用户配置 ==== #
MSD_FILE = Path(r"E:/固态组/LiLa2O3/势函数/LiLaO2/lammps/1900K/msd-1900.dat")  # 第一列 TimeStep，第二列 MSD (Å^2)
STRUCTURE_FILE = Path(r"E:/固态组/LiLa2O3/势函数/LiLaO2/CONTCAR-lammps.vasp")
TEMPERATURE_K = 1900.0
MOBILE_ION = "Li"
Z = 1.0                 # 价态
DT_FS = 2.0             # 每个 timestep 的 fs 长度
FIT_START_FRAC = 0.4    # 拟合起点比例
FIT_END_FRAC = 1.0      # 拟合终点比例
OUT_CSV = Path(r"E:/固态组/LiLa2O3/势函数/LiLaO2/lammps/1900K/msd_transport.csv")
# ================== #

E_CHARGE = 1.602176634e-19  # C
K_B_J = 1.380649e-23        # J/K


def main() -> None:
    if not MSD_FILE.exists():
        raise FileNotFoundError(f"MSD 文件不存在: {MSD_FILE}")
    if not STRUCTURE_FILE.exists():
        raise FileNotFoundError(f"结构文件不存在: {STRUCTURE_FILE}")

    data = pd.read_csv(MSD_FILE, sep=None, engine="python", header=None)
    data = data.dropna(axis=0, how="all").reset_index(drop=True)
    if data.shape[1] < 2:
        raise ValueError("MSD 文件需要至少两列: TimeStep, MSD")

    timestep = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy()
    msd_a2 = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy()
    valid = np.isfinite(timestep) & np.isfinite(msd_a2)
    timestep = timestep[valid]
    msd_a2 = msd_a2[valid]

    # 时间换算: timestep -> ps
    t_ps = timestep * DT_FS / 1000.0

    # 线性拟合区间
    n = len(t_ps)
    i0 = int(n * FIT_START_FRAC)
    i1 = int(n * FIT_END_FRAC)
    if i1 - i0 < 5:
        i0, i1 = 0, n

    k, b = np.polyfit(t_ps[i0:i1], msd_a2[i0:i1], 1)  # Å^2/ps
    d_a2_ps = k / 6.0
    d_cm2_s = d_a2_ps * 1e-4  # 1 Å^2/ps = 1e-4 cm^2/s

    structure = Structure.from_file(str(STRUCTURE_FILE))
    n_mobile = sum(1 for site in structure if site.specie.symbol == MOBILE_ION)
    if n_mobile == 0:
        raise ValueError(f"结构中找不到 {MOBILE_ION}")
    volume_cm3 = structure.volume * 1e-24
    n_cm3 = n_mobile / volume_cm3

    sigma_s_cm = n_cm3 * (Z * E_CHARGE) ** 2 * d_cm2_s / (K_B_J * TEMPERATURE_K)

    df = pd.DataFrame([
        {
            "temperature_K": TEMPERATURE_K,
            "slope_A2_per_ps": k,
            "D_cm2_per_s": d_cm2_s,
            "sigma_S_per_cm": sigma_s_cm,
            "sigma_mS_per_cm": sigma_s_cm * 1e3,
            "fit_start_idx": i0,
            "fit_end_idx": i1,
        }
    ])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"拟合公式: MSD(Å^2) = {k:.6e} * t(ps) + {b:.6e}")
    print(df.to_string(index=False))
    print(f"已写入 {OUT_CSV}")


if __name__ == "__main__":
    main()