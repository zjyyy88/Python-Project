#!/bin/bash

# 脚本功能：遍历所有 *-LiFSI-ads-** 文件夹，将其中的 CONTCAR 和 POSCAR 复制到上级目录并重命名，最后打包成 ads.zip

# 1. 遍历所有匹配 *-LiFSI-ads-** 的文件夹
for dir in */; do
  # 移除路径末尾的斜杠，得到干净的目录名 (例如: *-LiFSI-ads-*123)
  dir_name="${dir%/}"

  # 检查目标文件是否存在
  if [[ -f "$dir/CONTCAR" ]]; then
    # 复制 CONTCAR 到上一级目录并重命名
    cp "$dir/CONTCAR" "./${dir_name}-CONTCAR"
    echo "已复制: $dir/CONTCAR -> ./${dir_name}-CONTCAR"
  else
    echo "警告: 在目录 $dir 中未找到 CONTCAR 文件，跳过。"
  fi

  if [[ -f "$dir/POSCAR" ]]; then
    # 复制 POSCAR 到上一级目录并重命名
    cp "$dir/POSCAR" "./${dir_name}-POSCAR"
    echo "已复制: $dir/POSCAR -> ./${dir_name}-POSCAR"
  else
    echo "警告: 在目录 $dir 中未找到 POSCAR 文件，跳过。"
  fi
done

# 2. 将所有重命名的文件打包成 ads.zip
# 使用 ls 和 grep 来匹配我们刚刚创建的文件模式，防止打包原始文件或其他文件
echo "正在打包文件至 ads.zip ..."
zip -r ads.zip *-CONTCAR *-POSCAR

# 检查 zip 命令是否成功执行
if [[ $? -eq 0 ]]; then
  echo "成功创建压缩包: ads.zip"

  # 3. (可选) 清理：删除临时复制的文件
  read -p "是否要删除刚刚复制的 CONTCAR 和 POSCAR 文件? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f *-CONTCAR *-POSCAR
    echo "临时文件已删除。"
  else
    echo "临时文件保留在当前目录。"
  fi

else
  echo "错误: 创建压缩包失败！请检查是否有文件可以打包。"
fi
