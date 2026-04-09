# 1. 导入所有依赖模块
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PDEntry
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt

# 2. 构建体系所有相的 PDEntry（成分 + 每个化学式对应的总能量，单位 eV）
entries = [
    PDEntry(Composition("La2O3"), -41.944826),
    PDEntry(Composition("La"), -4.895479),
    PDEntry(Composition("Li2O"), -14.314615),
    PDEntry(Composition("La32Li1O48"), -670.83992),
    PDEntry(Composition("La32Li2O48"), -672.69114124),
    PDEntry(Composition("La32Li3O48"), -674.42556927),
    PDEntry(Composition("La32Li6O48"), -678.53968474),
    PDEntry(Composition("La32Li12O48"), -684.4147654),
    
    # ② 化合物相（VASP 计算得到的晶胞总能，单位 eV）
    PDEntry(Composition("La32Li14O48"), -685.68270599),
    PDEntry(Composition("La32Li15O48"), -683.1274741),
    PDEntry(Composition("La32Li16O48"), -661.95986939),
]

# 3. 构建相图对象（以 Li2O-La2O3-La 为端点）
terminal_compositions = [
    Composition("Li2O"),
    Composition("La2O3"),
    Composition("La"),
]
pd = CompoundPhaseDiagram(entries, terminal_compositions)

# 4. 提取凸包稳定相（可选，验证用）
stable_entries = pd.stable_entries
print("✅ 凸包上的稳定相：")
for entry in stable_entries:
    print(f"成分：{entry.original_entry.composition}，单原子能量：{entry.energy_per_atom:.4f} eV/atom")

# ✅ 批量输出指定化学式的凸包能（eV/atom）
target_formulas = [
    "La32Li1O48",
    "La32Li2O48",
    "La32Li3O48",
    "La32Li6O48",
    "La32Li12O48",
    "La32Li14O48",
    "La32Li15O48",
    "La32Li16O48",
]

print("=" * 60)
print("✅ 指定化学式的凸包能 ΔE_hull (eV/atom)：")
for formula in target_formulas:
    normalized_formula = Composition(formula).formula
    matched_entry = next(
        (te for te in pd.all_entries if te.original_entry.composition.formula == normalized_formula),
        None,
    )
    if matched_entry is None:
        print(f"{formula}: 未找到对应条目")
        continue

    delta_e_hull = pd.get_e_above_hull(matched_entry)
    print(f"{formula}: ΔE_hull = {delta_e_hull:.4f} eV/atom")

# 5. 绘制凸包相图（无修改，正常运行）
plotter = PDPlotter(pd, show_unstable=True)  # show_unstable=True 显示亚稳相
plotter.show()
plt.title("体系 0K 凸包相图", fontsize=12)
plt.xlabel("O成分占比", fontsize=10)
plt.ylabel("形成能 (eV/atom)", fontsize=10)
plt.show()