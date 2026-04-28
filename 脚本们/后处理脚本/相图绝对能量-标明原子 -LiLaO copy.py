# 1. 导入所有依赖模块
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PDEntry
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt

# 2. 构建体系所有相的 PDEntry（成分 + 每个化学式对应的总能量，单位 eV）
entries = [
    PDEntry(Composition("Li2S"), -12.235447),
    PDEntry(Composition("LiCl"),-7.834113),
    PDEntry(Composition("LiI"), -6.152928),
    PDEntry(Composition("P2S5"), -35.395373),
    PDEntry(Composition("Li48P8S40Cl7I1"), -464.6808073),
    

    
    # ② 化合物相（VASP 计算得到的晶胞总能，单位 eV）
]

# 3. 构建相图对象（以 Li2O-La2O3-La 为端点）
terminal_compositions = [
    Composition("Li2S"),
    Composition("LiCl"),
    Composition("LiI"),
    Composition("P2S5"),
]
pd = CompoundPhaseDiagram(entries, terminal_compositions)

# 4. 提取凸包稳定相（可选，验证用）
stable_entries = pd.stable_entries
print("✅ 凸包上的稳定相：")
for entry in stable_entries:
    print(f"成分：{entry.original_entry.composition}，单原子能量：{entry.energy_per_atom:.4f} eV/atom")

# ✅ 批量输出指定化学式的凸包能（eV/atom）
target_formulas = [   
"Li48P8S40Cl7I1",

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