# 1. 导入所有依赖模块
#安装pymatgen,mp-api,终端联网
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from mp_api.client import MPRester

# 1. 导入所有依赖模块
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from mp_api.client import MPRester

# 2. 初始化 API 接口
# 建议使用 with 语句自动管理连接
with MPRester("HiDNgsh0Zz5RnMjLIjH3HxfFjTwnRIgb") as mpr:

    # 3. 定义查询的化学体系
    # 例如：查询 Li-In-Cl 体系的所有条目
    chemsys = ["Li", "Y", "Bi","Cl"]  # 修改为您需要的体系，如 ['Li', 'P', 'S', 'Cl']
    #print(f"正在下载 {chemsys} 体系数据...")
    entries = mpr.get_entries_in_chemsys(chemsys)  # 获取该体系下的所有化合物数据
    #print(f"已下载 {len(entries)} 个条目")

    # 定义锂金属参考能量（手动设定值，也可从 entries 自动获取）
    Li_potential = -1.92  

    # 4. 构建相图对象
    PD = PhaseDiagram(entries=entries)

    # 5. 计算相对于 Li 的电压曲线 (Element Profile)
    # get_element_profile 计算随 Li 化学势变化的平衡产物
    stages = PD.get_element_profile("Li", Composition("Li3Y0.66Bi0.33Cl6"))

    # 6. 打印每个阶段的结果
    # stage['chempot'] 是该阶段的 Li 化学势
    # 电压 = -(μ_Li - μ_Li0) = -(chempot - Li_potential)
    # 注意：这里您的公式是用 Li_potential - chempot，这实际上也是电压（假设 Li_potential 是负值且更小）
    # 但相图输出的 chempot 是绝对能量，Li_potential 也是参考绝对能量。
    # 标准电压公式 V = -(μ_Li_sample - μ_Li_metal)
    print(f"{'电压(V)':<10}\t{'反应产物'}\t{'临界组分'}")
    for stage in stages:
        voltage = -(stage['chempot'] - Li_potential) # 计算电压
        print(f"{voltage:.4f}", stage['reaction'], stage['critical_composition'], sep='\t')
 
        