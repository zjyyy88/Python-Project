# 1. 导入所有依赖模块
from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt

# 2. 构建体系所有相的PDEntry（成分+单原子能量，eV/atom）:写入方法可以是所有原子数的能量，不用表明原子数。
entries = [
    PDEntry(Composition("Li"), -1.3,1),   # Li单质晶胞：总绝对能-105.26 eV，含16个Li原子
    PDEntry(Composition("Bi"),  -1.36,1),    # Y单质晶胞：总绝对能-328.58 eV，含2个Y原子
    PDEntry(Composition("Y"), -2.24,1),
     PDEntry(Composition("Cl"), -0.23,1),
    PDEntry(Composition("LiCl"), -7.3857361,2),   # Li单质晶胞：总绝对能-105.26 eV，含16个Li原子
    PDEntry(Composition("BiCl3"),  -14.32141014,4),    # Y单质晶胞：总绝对能-328.58 eV，含2个Y原子
    PDEntry(Composition("YCl3"), -21.17910362,4),    # Cl单质晶胞：总绝对能-226.15 eV，含8个Cl原子
    
    # ② 化合物相（你的研究体系，VASP计算的晶胞总绝对能量+原子数）
    #PDEntry(Composition("Li9Y3Cl18"), -129.5926164,30),  # Li3YCl6晶胞：总绝对能-2156.82 eV，含20个原子
    PDEntry(Composition("Li9Y2Bi1Cl18"),  -121.4773016,30),  # LiYCl4晶胞：总绝对能-1428.36 eV，含12个原子
   
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
target_comp = Composition("Y2Bi1Cl18Li9")
# 从entries中提取LiYCl4的「晶胞总绝对能」和「原子数」
target_total_energy = -121.4773016
target_n_atoms = 30
# 转换为单原子绝对能量（仅这一步简单除法，Pymatgen需要）
target_e_per_atom = target_total_energy / target_n_atoms

# ✅ 计算凸包能（eV/atom），判定稳定/亚稳
delta_e_hull = pd.get_e_above_hull(PDEntry(target_comp, target_e_per_atom))
print("="*60)
print(f"✅ 目标相 {target_comp} 的凸包能ΔE_hull = {delta_e_hull:.4f} eV/atom")
if delta_e_hull == 0:
    print(f"✅ 结论：{target_comp} 是【热力学稳定相】（在凸包上）")
elif delta_e_hull > 0:
    print(f"✅ 结论：{target_comp} 是【亚稳相】（ΔE_hull>0，可合成）")

# 5. 绘制凸包相图（无修改，正常运行）
plotter = PDPlotter(pd, show_unstable=True)  # show_unstable=True 显示亚稳相
plotter.show()
plt.title("体系 0K 凸包相图", fontsize=12)
plt.xlabel("O成分占比", fontsize=10)
plt.ylabel("形成能 (eV/atom)", fontsize=10)
plt.show()