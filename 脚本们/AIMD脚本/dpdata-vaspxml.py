import dpdata

# 从单个 OUTCAR 文件读取
system = dpdata.LabeledSystem("vasprun.xml", fmt="vasp/xml")

# 转换为 DeepMD 格式
system.to("deepmd/npy", "deepmd_data", set_size=system.get_nframes())

print(f"转换完成！")
print(f"总帧数: {system.get_nframes()}")
print(f"原子数: {system.get_natoms()}")
print(f"原子类型: {system['atom_names']}")
