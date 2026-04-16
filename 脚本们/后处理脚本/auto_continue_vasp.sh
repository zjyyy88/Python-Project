#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# 未收敛判据（VASP 几何优化收敛标志）
TARGET_LINE="reached required accuracy - stopping structural energy minimisation"
JOB_SCRIPT="vasp6.5.1_oneapi2025.1.0_vtst_pbs.sh"
JOB_DIR_PATTERN="./*/jobs/*/"

job_dirs=($JOB_DIR_PATTERN)
if [[ ${#job_dirs[@]} -eq 0 ]]; then
  echo "未匹配到目录: $JOB_DIR_PATTERN"
  exit 1
fi

for dir in "${job_dirs[@]}"; do
  echo "=============================="
  echo "处理目录: $dir"

  outcat_file="${dir}OUTCAT"
  outcar_file="${dir}OUTCAR"
  contcar_file="${dir}CONTCAR"
  poscar_file="${dir}POSCAR"
  job_file="${dir}${JOB_SCRIPT}"

  if [[ -f "$outcat_file" ]]; then
    out_file="$outcat_file"
  elif [[ -f "$outcar_file" ]]; then
    out_file="$outcar_file"
  else
    echo "跳过：未找到 OUTCAT/OUTCAR"
    continue
  fi

  if grep -Fq "$TARGET_LINE" "$out_file"; then
    echo "已收敛，不续算。"
    continue
  fi

  echo "未检测到收敛标志，准备续算。"

  if [[ ! -s "$contcar_file" ]]; then
    echo "跳过：CONTCAR 不存在或为空。"
    continue
  fi

  if [[ -f "$poscar_file" ]]; then
    cp -f "$poscar_file" "${poscar_file}.bak.$(date +%Y%m%d_%H%M%S)"
  fi

  mv -f "$contcar_file" "$poscar_file"
  echo "已执行: mv CONTCAR POSCAR"

  if [[ ! -f "$job_file" ]]; then
    echo "跳过：未找到提交脚本 $JOB_SCRIPT"
    continue
  fi

  (
    cd "$dir"
    echo "提交作业: qsub $JOB_SCRIPT"
    qsub "$JOB_SCRIPT"
  )
done

echo "批量检查完成。"
