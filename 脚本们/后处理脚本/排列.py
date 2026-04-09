import pandas as pd
from io import StringIO

data = """after				before			
band center	s-band(eV)	p-band(eV)	d-band(eV)	band center	s-band(eV)	p-band(eV)	d-band(eV)
111-surface-Al_CONTCAR	0.238	2.706	0	111-surface-Ba_CONTCAR	-24.308	-10.156	6.183
111-surface-Ba_CONTCAR	-24.43	-10.394	5.847	111-surface-Ca_CONTCAR	-37.373	-18.709	6.372
111-surface-Bi_CONTCAR	-9.167	1.083	5.862	111-surface-Co_CONTCAR	-0.959	-0.365	-0.876
111-surface-Bi_d_CONTCAR	-9.14	1.139	5.718	111-surface-Cu_CONTCAR	1.114	1.672	-1.632
111-surface-Ca_CONTCAR	4.205	-17.335	6.359	111-surface-Fe3+_CONTCAR	1.202	1.195	-1.775
111-surface-Ce__CONTCAR	-34.125	-17.024	3.542	111-surface-Mn_CONTCAR	1.034	-44.024	-0.527
111-surface-Co_CONTCAR	0.239	0.02	-1.316	111-surface-Ni_CONTCAR	0.674	0.567	-1.562
111-surface-Cr_CONTCAR	-0.759	-1.213	-0.365	111-surface-Sr_CONTCAR	-31.735	-13.967	4.56
111-surface-Cu_CONTCAR	1.289	1.85	-2.446	111-surface-W_CONTCAR	-53.263	-36.13	-0.233
111-surface-Fe3+_CONTCAR	1.416	1.546	-1.743	111-surface-Zn_CONTCAR	-2.947	1.125	-19.076
111-surface-Ga_CONTCAR	-2.736	1.922	1.844	111-surface-Ce_	-30.377	-15.931	3.526
111-surface-Gd_CONTCAR	185.564	185.437	14.754	111-surface-Cr_	0.75	-39.91	-0.309
111-surface-Hf_CONTCAR	-1.641	-1.971	-0.244	111-surface-Ga_	-1.876	2.804	-13.832
111-surface-Mn_CONTCAR	-0.133	-0.153	-0.632	111-surface-Gd_	6.95	-17.85	3.396
111-surface-Nd_CONTCAR	-33.402	-15.449	0.971	111-surface-Hf_	-1.461	-28.389	0.053
111-surface-Ni_CONTCAR	-1.502	-1.261	-0.42	111-surface-Nd_	-33.476	-17.373	3.522
111-surface-Pb4+_CONTCAR	-5.96	2.617	6.746	111-surface-Pb_	-5.342	3.584	-15.402
111-surface-Pb_CONTCAR	-5.783	2.828	6.034	111-surface-Sb_	-7	1.216	5.984
111-surface-Sb_CONTCAR	-6.987	1.213	6.148	111-surface-Sc_	-39.349	-25.743	2.858
111-surface-Sc_CONTCAR	-0.177	4.029	2.546	111-surface-Sm_	-33.761	-18.67	3.193
111-surface-Sn_CONTCAR	-524.785	-552.343	-564.576	111-surface-Sn_	-4.509	2.259	-20.482
111-surface-Sr_CONTCAR	-32.189	-14.233	6.733	111-surface-U_	-40.829	-18.422	1.647
111-surface-U_CONTCAR	-40.501	-18.491	1.178	111-surface-Zr_			
111-surface-W-new_CONTCAR	-1.434	-1.916	-0.631	111-surface-Al_CONTCAR	0.225	2.579	0
111-surface-W-tiaomag_CONTCAR	1.193	2.885	12.371	111-surface-Bi_CONTCAR	-8.915	1.572	-21.928
111-surface-Y_CONTCAR	-36.033	-18.098	3.624	111-surface-Lu_CONTCAR	1.149	-21.789	3.857
111-surface-Zn_CONTCAR	0.646	3.479	-5.654	111-surface-In_CONTCAR	-1.775	3.29	-13.74
111-surface-Zr_CONTCAR	-44.787	-27.811	0.178	111-surface-Y_CONTCAR	-36.341	-18.087	3.557
					"""

# 修正列名，明确8列结构（after和before各4列）
columns = [
    'after_name', 'after_s', 'after_p', 'after_d',
    'before_name', 'before_s', 'before_p', 'before_d'
]

# 解析数据，跳过原始表头行（因已手动指定columns）
df = pd.read_csv(StringIO(data), sep='\s+', names=columns, skiprows=2)  # skiprows=2跳过前两行说明文字

# 提取after组并清洗文件名
after = df[['after_name', 'after_s', 'after_p', 'after_d']].dropna(subset=['after_name']).rename(columns={'after_name': 'name'})

# 提取before组并清洗文件名
before = df[['before_name', 'before_s', 'before_p', 'before_d']].dropna(subset=['before_name']).rename(columns={'before_name': 'name'})

# 优化文件名清洗函数
def clean_name(name):
    name = name.strip()
    if '_CONTCAR' not in name:
        name = name.rstrip('_') + '_CONTCAR'
    else:
        name = name.replace('__', '_')
    return name

after['name'] = after['name'].apply(clean_name)
before['name'] = before['name'].apply(clean_name)

# 合并数据并排序
merged = pd.merge(after, before, on='name', how='outer').sort_values('name').fillna('')

# 输出结果
print("合并后的数据：")
print(merged)
merged.to_excel('aligned_data.xlsx', index=False)