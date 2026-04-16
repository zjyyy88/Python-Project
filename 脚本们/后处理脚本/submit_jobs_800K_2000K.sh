#!/usr/bin/env bash
set -euo pipefail

# Batch submit VASP jobs for 800K to 2000K (step 200K)
# Run this script in the directory where source files exist:
# INCAR KPOINTS POTCAR CONTCAR run.sh zjy.sh

required_files=(INCAR KPOINTS POTCAR CONTCAR run.sh zjy.sh)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing required file: $f" >&2
    exit 1
  fi
done

for temp in $(seq 800 200 2000); do
  dir="${temp}K"
  mkdir -p "$dir"

  cp -f INCAR KPOINTS POTCAR CONTCAR run.sh zjy.sh "$dir/"

  temp_from_dir="${dir%K}"

  # Replace literal 300 in INCAR with folder temperature.
  sed -i "s/\b300\b/${temp_from_dir}/g" "$dir/INCAR"

  echo "[INFO] Prepared $dir (INCAR: 300 -> ${temp_from_dir})"

  (
    cd "$dir"
    chmod +x zjy.sh
    ./zjy.sh
  )
done

echo "[DONE] All jobs submitted from 800K to 2000K."
