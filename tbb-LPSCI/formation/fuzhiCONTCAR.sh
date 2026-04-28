
%oldrs_interactive.sh

echo "=== 交互式CONTCAR文件复制 ==="

# 查找所有包含CONTCAR的文件夹
contcar_folders=($(find ./jobs2/*/ -name "CONTCAR" -type f -exec dirname {} \; | sort -u))

if [ ${#contcar_folders[@]} -eq 0 ]; then
    echo "❌ 未找到任何包含CONTCAR文件的文件夹"
    exit 1
fi

echo "找到 ${#contcar_folders[@]} 个包含CONTCAR文件的文件夹:"
for i in "${!contcar_folders[@]}"; do
    folder_name=$(basename "${contcar_folders[i]}")
    echo "  [$((i+1))] $folder_name"
done

echo ""
read -p "是否复制所有CONTCAR文件? (y/N): " confirm

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "操作取消"
    exit 0
fi

# 执行复制
for folder in "${contcar_folders[@]}"; do
    folder_name=$(basename "$folder")
    target_file="${folder_name}_CONTCAR"
    
    if cp "$folder/CONTCAR" "$target_file" 2>/dev/null; then
        echo "✅ 复制: $target_file"
    else
        echo "❌ 失败: $target_file"
    fi
done

echo "复制完成"chk=Li_EC_Opt.chk
