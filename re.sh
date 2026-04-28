#!/bin/bash
# 文件名：energy_per_formula_gcd.sh
# 功能：计算每化学式能量 = 总能量 / 最大公约数

echo "=== 化学式能量分析（每化学式能量 = 总能量 / 最大公约数）==="

# 输出文件
OUTPUT_CSV="energy_per_formula_gcd.csv"
OUTPUT_TXT="energy_summary_gcd.txt"

# 创建输出文件头
echo "文件夹名称,化学式,总能量(eV),总原子数,最大公约数,每化学式能量(eV/formula)" > "$OUTPUT_CSV"
echo "化学式能量分析报告 - 生成时间: $(date)" > "$OUTPUT_TXT"
echo "==========================================================" >> "$OUTPUT_TXT"

# 查找所有POSCAR文件
echo "搜索POSCAR和OUTCAR文件..."
poscar_files=($(find ./*/*/metagga_relax -name "POSCAR" -type f | sort))

if [ ${#poscar_files[@]} -eq 0 ]; then
    echo "❌ 未找到任何POSCAR文件"
    exit 1
fi

echo "找到 ${#poscar_files[@]} 个结构"
echo ""

# 函数：计算最大公约数
calculate_gcd() {
    local a=$1
    local b=$2
    while [ $b -ne 0 ]; do
        local temp=$b
        b=$((a % b))
        a=$temp
    done
    echo $a
}

# 函数：计算数组的最大公约数
calculate_gcd_array() {
    local arr=("$@")
    local gcd=${arr[0]}
    
    for i in "${!arr[@]}"; do
        if [ $i -eq 0 ]; then
            continue
        fi
        gcd=$(calculate_gcd $gcd ${arr[$i]})
    done
    
    echo $gcd
}

# 处理每个结构
total_count=0
valid_count=0

for poscar_file in "${poscar_files[@]}"; do
    total_count=$((total_count + 1))
    dir_path=$(dirname "$poscar_file")
    dir_name=$(basename "$dir_path")
    
    echo "分析 [$total_count]: $dir_name"
    
    # 检查OUTCAR文件是否存在
    outcar_file="$dir_path/OUTCAR"
    if [ ! -f "$outcar_file" ]; then
        echo "  ❌ 缺少OUTCAR文件，跳过"
        continue
    fi
    
    # 检查计算是否收敛
    if ! grep -q "reached required accuracy" "$outcar_file"; then
        echo "  ⚠️  计算未收敛，跳过"
        continue
    fi
    
    # 提取能量
    energy_line=$(grep "TOTEN" "$outcar_file" | tail -1)
    if [ -z "$energy_line" ]; then
        echo "  ❌ 无法提取能量，跳过"
        continue
    fi
    
    energy=$(echo "$energy_line" | awk '{print $5}')
    if [ -z "$energy" ]; then
        echo "  ❌ 能量提取失败，跳过"
        continue
    fi
    
    # 提取化学式
    elements_line=$(sed -n '6p' "$poscar_file" 2>/dev/null | xargs)
    counts_line=$(sed -n '7p' "$poscar_file" 2>/dev/null | xargs)
    
    if [ -z "$elements_line" ] || [ -z "$counts_line" ]; then
        echo "  ❌ 无效的POSCAR格式，跳过"
        continue
    fi
    
    # 解析元素和原子数
    elements=($elements_line)
    atom_counts=($counts_line)
    
    # 生成化学式
    formula=""
    total_atoms=0
    
    for i in "${!elements[@]}"; do
        formula+="${elements[i]}${atom_counts[i]}"
        total_atoms=$((total_atoms + atom_counts[i]))
    done
    
    valid_count=$((valid_count + 1))
    
    # 计算最大公约数
    gcd=$(calculate_gcd_array "${atom_counts[@]}")
    
    # 计算每化学式能量：总能量 / 最大公约数
    energy_per_formula=$(echo "scale=6; $energy / $gcd" | bc -l 2>/dev/null || echo "N/A")
    
    echo "  ✅ 文件夹: $dir_name"
    echo "     化学式: $formula"
    echo "     总能量: $energy eV"
    echo "     总原子数: $total_atoms"
    echo "     原子数数组: ${atom_counts[*]}"
    echo "     最大公约数: $gcd"
    echo "     每化学式能量: $energy_per_formula eV/formula (总能量 / 最大公约数)"
    
    # 写入CSV
    echo "$dir_name,$formula,$energy,$total_atoms,$gcd,$energy_per_formula" >> "$OUTPUT_CSV"
    
    # 写入文本文件
    echo "文件夹: $dir_name" >> "$OUTPUT_TXT"
    echo "化学式: $formula" >> "$OUTPUT_TXT"
    echo "总能量: $energy eV" >> "$OUTPUT_TXT"
    echo "总原子数: $total_atoms" >> "$OUTPUT_TXT"
    echo "原子数数组: ${atom_counts[*]}" >> "$OUTPUT_TXT"
    echo "最大公约数: $gcd" >> "$OUTPUT_TXT"
    echo "每化学式能量: $energy_per_formula eV/formula (总能量 / 最大公约数)" >> "$OUTPUT_TXT"
    echo "---" >> "$OUTPUT_TXT"
    
    echo ""
done

echo ""
echo "=== 分析完成 ==="
echo "总共处理: $total_count 个结构"
echo "有效结构: $valid_count 个"
echo ""
echo "输出文件:"
echo "  - $OUTPUT_CSV (详细能量数据)"
echo "  - $OUTPUT_TXT (文本报告)"

# 显示结果汇总
echo ""
echo "=== 结果汇总（按文件夹顺序）==="
printf "%-25s %-20s %-10s %-8s %-12s\n" "文件夹名称" "化学式" "总原子数" "GCD" "每化学式(eV)"
echo "======================================================================"

tail -n +2 "$OUTPUT_CSV" | while IFS=',' read -r folder formula energy total_atoms gcd per_formula; do
    printf "%-25s %-20s %-10d %-8d %-12.6f\n" "$folder" "$formula" "$total_atoms" "$gcd" "$per_formula"
done
