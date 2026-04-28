#!/bin/bash

# ====================== 变量定义 ======================
source_dir="/public/home/catl-bpit-hpc55/calc/TBB/formation"  # 源文件路径（含POSCAR、INCAR等）
potcar_base="/public/home/catl-bpit-hpc55/calc/TBB/formation"                         # POTCAR文件的基础路径
target_base="$source_dir/jobs"                                         # 目标文件夹的基路径（所有任务文件夹将在此下）

# ====================== 创建目标基路径 ======================
mkdir -p "$target_base"

# ====================== 遍历所有POSCAR文件 ======================
for poscar_file in "$source_dir"/*.poscar; do
    # 提取文件名（不含路径），例如 "POSCAR-Cu" → "Cu"
    filename=$(basename "$poscar_file")
    element=${filename%.poscar}  # 去掉后缀 ".vasp", 得到元素名（如 "Cu"）
    
    # 创建元素对应的文件夹（例如 "Cu"）
    element_dir="$target_base/$element"
    mkdir -p "$element_dir"
    
    # --------------------- 步骤1：移动并重命名POSCAR ---------------------
    cp "$poscar_file" "$element_dir/POSCAR"  # 复制（或用mv移动）POSCAR到目标文件夹，重命名为POSCAR
    
    # --------------------- 步骤2：合并POTCAR ---------------------
    potcar_file="$element_dir/POTCAR"
    > "$potcar_file"  # 清空或创建POTCAR文件
    
#    # 读取POSCAR的第6行（元素行，格式如 "Cu" 或 "Cu Fe O"）
    elements_line=$(sed -n '6p' "$element_dir/POSCAR")
    IFS=' ' read -ra elements <<< "$elements_line"  # 按空格分割元素列表
    
    # 遍历每个元素，提取对应的POTCAR并合并
    for e in "${elements[@]}"; do
        potcar_source="$potcar_base/POTCAR-$e"  # 源POTCAR路径（如 "/home/.../POTCAR-Cu"）
        if [ -f "$potcar_source" ]; then        # 检查文件是否存在
            cat "$potcar_source" >> "$potcar_file"  # 追加到目标POTCAR
        else
            echo "警告：未找到POTCAR文件 $potcar_source，跳过该元素。"
        fi
    done
    
    # --------------------- 步骤3：复制配置文件 ---------------------
    cp "$source_dir/INCAR" "$element_dir/"
    cp "$source_dir/KPOINTS" "$element_dir/"
#    cp "$source_dir/POTCAR" "$element_dir/"
    cp "$source_dir/vasp6.5.1_oneapi2025.1.0_vtst_pbs.sh" "$element_dir/"
done
#    cd "$element_dir"
#    qsub vasp5-4-4.slurm  # 提交VASP任务
#    cd - > /dev/null      # 返回原路径，避免影响后续操作
     cd /public/home/catl-bpit-hpc55/calc/TBB/formation/jobs/ && for d in */; do (cd "$d" && qsub vasp6.5.1_oneapi2025.1.0_vtst_pbs.sh); done
echo "所有任务处理完成！"
