#!/bin/bash

# 输出文件
output_file="energies.txt"

# 清空或创建输出文件
echo -e "Folder\tEnergy(eV)" > "$output_file"

# 遍历所有  文件夹
for folder in *; do
    if [ -d "$folder" ]; then
        # 检查 OUTCAR 是否存在
        if [ -f "$folder/OUTCAR" ]; then
            # 提取最终能量 (最后一个 'free  energy   TOTEN' 行)
            energy=$(grep "free  energy   TOTEN" "$folder/OUTCAR" | tail -n 1 | awk '{print $5}')
            # 写入输出文件
            echo -e "$folder\t$energy" >> "$output_file"
        else
            echo "Warning: $folder/OUTCAR not found!" >&2
        fi
    fi
done

echo "Energy extraction completed. Results saved to $output_file."
