# 1. 导入所有依赖模块
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PDEntry
from pymatgen.analysis.phase_diagram import PDPlotter
import matplotlib.pyplot as plt

# 2. 构建体系所有相的 PDEntry（成分 + 每个化学式对应的总能量，单位 eV）
entries = [
    PDEntry(Composition("La2O3"), -96.05805),
    PDEntry(Composition("La"),-29.741277),
    PDEntry(Composition("Li2O"), -16.836728),
    PDEntry(Composition("La32O48"), -1536.803308),
    PDEntry(Composition("Li1La32O48"), -1538.3256),
    PDEntry(Composition("Li2La32O48"), -1540.2683),
    PDEntry(Composition("Li3La32O48"), -1542.1830),
    PDEntry(Composition("Li4La32O48"), -1543.4968),
    PDEntry(Composition("Li5La32O48"), -1545.0632),
    PDEntry(Composition("Li6La32O48"), -1547.1499),
    PDEntry(Composition("Li7La32O48"), -1548.5724),
    PDEntry(Composition("Li8La32O48"), -1549.7982),
    PDEntry(Composition("Li9La32O48"), -1551.4277),
    PDEntry(Composition("Li10La32O48"), -1553.927),
    PDEntry(Composition("Li11La32O48"), -1555.302),
    PDEntry(Composition("Li12La32O48"), -1557.243),
    PDEntry(Composition("Li13La32O48"), -1559.079),
    PDEntry(Composition("Li14La32O48"), -1561.557),
    PDEntry(Composition("Li15La32O48"), -1563.067),
    PDEntry(Composition("Li16La32O48"), -1564.558),
    PDEntry(Composition("Li17La32O48"), -1566.943),
    PDEntry(Composition("Li18La32O48"), -1569.225),
    PDEntry(Composition("Li19La32O48"), -1571.329),
    PDEntry(Composition("Li20La32O48"), -1573.296),
    PDEntry(Composition("Li21La32O48"), -1574.671),
    PDEntry(Composition("Li22La32O48"), -1576.969),
    PDEntry(Composition("Li24La32O48"), -1566.481),

    
    # ② 化合物相（VASP 计算得到的晶胞总能，单位 eV）
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
"Li1La32O48",
"Li2La32O48",
"Li3La32O48",
"Li4La32O48",
"Li5La32O48",
"Li6La32O48",
"Li7La32O48",
"Li8La32O48",
"Li9La32O48",
"Li10La32O48",
"Li11La32O48",
"Li12La32O48",
"Li13La32O48",
"Li14La32O48",
"Li15La32O48",
"Li16La32O48",
"Li17La32O48",
"Li18La32O48",
"Li19La32O48",
"Li20La32O48",
"Li21La32O48",
"Li22La32O48",
"Li24La32O48"
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