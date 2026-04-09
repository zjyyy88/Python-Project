# 1. 导入所有依赖模块
from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt

# 2. 构建体系所有相的PDEntry（成分+单原子能量，eV/atom）
'''
entries = [
    PDEntry(Composition("Li"), 0.0),    # Li单质，参考态能量0
    PDEntry(Composition("O2"), 0.0),    # O2单质，参考态能量0
    PDEntry(Composition("Li2O"), -1.5), # Li2O 单原子能量
    PDEntry(Composition("Li2O2"), -1.2),# Li2O2 单原子能量
    PDEntry(Composition("LiO2"), -0.8)  # LiO2 单原子能量
]
'''
entries = [
    PDEntry(Composition("Li"), 0),    # Li单质，参考态能量0
    PDEntry(Composition("O2"), 0),    # O2单质，参考态能量0
    PDEntry(Composition("Li2O"), -1.5), # Li2O 单原子能量
    PDEntry(Composition("Li2O2"), -1.2),# Li2O2 单原子能量
    PDEntry(Composition("LiO2"), -0.8)  # LiO2 单原子能量
]

# 3. 构建相图对象
pd = PhaseDiagram(entries)

# 4. 提取凸包稳定相（可选，验证用）
stable_entries = pd.stable_entries
print("✅ 凸包上的稳定相：")
for entry in stable_entries:
    print(f"成分：{entry.composition}，单原子能量：{entry.energy_per_atom:.2f} eV/atom")

# ✅ 关键修复：新版Pymatgen 计算凸包能的正确写法
# 分2步：① 定义目标相成分 ② 传入「成分+目标相单原子能量」计算凸包能
target_comp = Composition("LiO2")  # 目标相成分
target_energy_per_atom = -0.8      # 目标相的单原子能量（和上面entries里一致）
# 计算凸包能 ΔE_hull（eV/atom）
delta_e_hull = pd.get_e_above_hull(PDEntry(target_comp, target_energy_per_atom))
print(f"\n✅ LiO2的凸包能（ΔE_hull）：{delta_e_hull:.4f} eV/atom")

# 5. 绘制凸包相图（无修改，正常运行）
plotter = PDPlotter(pd, show_unstable=True)  # show_unstable=True 显示亚稳相
plotter.show()
plt.title("Li-O体系 0K 凸包相图", fontsize=12)
plt.xlabel("O成分占比", fontsize=10)
plt.ylabel("形成能 (eV/atom)", fontsize=10)
plt.show()