# ===================== 终极版：论文级输出 + 数据导出 + 样式优化 =====================
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from pymatgen.io.vasp import Vasprun
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------- 1. 参数配置 --------------------------
OUTCAR_PATH = r"c:\Users\ZHANGJY02\Desktop\vasprun.xml"
TARGET_ION = "Li"
T = 1600  # K
FIT_RANGE = (1000, 5000)  # 拟合区间

# -------------------------- 2. 轨迹读取与分析 --------------------------
vasprun = Vasprun(OUTCAR_PATH, parse_dos=False, parse_eigen=False)
#diff = DiffusionAnalyzer(vasprun.get_ionic_trajectory(), specie=TARGET_ION, temperature=T)
diff = DiffusionAnalyzer(vasprun.get_trajectory(), specie=TARGET_ION, temperature=T)

# -------------------------- 3. 结果计算 --------------------------
D = diff.diffusion_coeff  # 扩散系数
sigma = diff.conductivity  # 电导率
msd_time = diff.msd[:, 0]  # MSD时间轴 (ps)
msd_value = diff.msd[:, 1]  # MSD位移轴 (Å²)
fit_line = diff.msd_fit  # 拟合直线数据
# 计算x/y/z方向的MSD
msd_x = diff.get_msd(component="x")
msd_y = diff.get_msd(component="y")
msd_z = diff.get_msd(component="z")

# 绘制分方向MSD曲线
plt.plot(msd_time, msd_x, label="MSD-X")
plt.plot(msd_time, msd_y, label="MSD-Y")
plt.plot(msd_time, msd_z, label="MSD-Z")
plt.legend()
plt.show()
# -------------------------- 4. 导出MSD数据为CSV（便于Origin二次绘图）--------------------------
msd_df = pd.DataFrame({
    "Time (ps)": msd_time,
    f"{TARGET_ION} MSD (Å²)": msd_value,
    "Fitting Line (Å²)": fit_line
})
msd_df.to_csv(f"{TARGET_ION}_MSD_data.csv", index=False)
print(f"✅ MSD数据已导出至：{TARGET_ION}_MSD_data.csv")

# -------------------------- 5. 论文级MSD曲线绘制 --------------------------
plt.rcParams["font.family"] = "Times New Roman"  # 论文字体
plt.figure(figsize=(7, 5), dpi=300)

# 绘制曲线
plt.plot(msd_time, msd_value, color="#2E86AB", linewidth=2.5, label="MSD")
plt.plot(msd_time, fit_line, color="#A23B72", linewidth=2, linestyle="--", label="Einstein Fitting")

# 标注拟合区间
plt.axvspan(FIT_RANGE[0], FIT_RANGE[1], alpha=0.1, color="gray", label=f"Fit Range: {FIT_RANGE[0]}-{FIT_RANGE[1]} ps")

# 坐标轴与标签优化
plt.xlabel("Time (ps)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Squared Displacement (Å²)", fontsize=12, fontweight="bold")
plt.title(f"{TARGET_ION}$^+$ Ion MSD at {T} K", fontsize=14, pad=15)
plt.legend(loc="upper left", fontsize=10, frameon=True, fancybox=True)
plt.grid(alpha=0.2, linestyle="-", linewidth=0.8)
plt.xlim(0, np.max(msd_time))
plt.ylim(0, np.max(msd_value)+0.5)

# 保存矢量图（pdf无失真，论文首选）
plt.savefig(f"{TARGET_ION}_MSD_{T}K.pdf", dpi=300, bbox_inches="tight", format="pdf")
plt.show()

# -------------------------- 6. 打印最终结果（论文格式）--------------------------
print("="*70)
print(f"📊 AIMD Diffusion Analysis Results (T = {T} K)")
print(f"🎯 Carrier Ion: {TARGET_ION}^+")
print(f"📈 Diffusion Coefficient (D): {D:.4e} cm² s⁻¹")
print(f"⚡ Ionic Conductivity (σ): {sigma:.4e} S cm⁻¹")
print(f"📐 Fitting Slope (6D): {np.polyfit(msd_time, fit_line, 1)[0]:.4f} Å² ps⁻¹")
print("="*70)