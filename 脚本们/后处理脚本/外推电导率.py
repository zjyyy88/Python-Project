import numpy as np
import pandas as pd

OUTDIR = r"E:/固态组/LiLa2O3/势函数/LiLaO2/transport_results_ucheck"  # 修改成你的目录
target_T = 300.0  # K
kB_eV = 8.617333262e-5  # eV/K

df = pd.read_csv(f"{OUTDIR}/transport_summary.csv")
T = df["temperature_K"].to_numpy()
sigma = df["sigma_S_per_cm"].to_numpy()

# Arrhenius 拟合 ln(σT) vs 1/T
x = 1.0 / T
y = np.log(sigma * T)
slope, intercept = np.polyfit(x, y, 1)  # slope ≈ -Ea/kB
Ea_eV = -slope * kB_eV
# 预测 300 K
pred_ln_sigmaT = intercept + slope * (1.0 / target_T)
pred_sigma = np.exp(pred_ln_sigmaT) / target_T  # σ = (σT)/T

print(f"Ea (from σT) = {Ea_eV:.4f} eV")
print(f"σ @ {target_T:.1f} K = {pred_sigma:.4e} S/cm")