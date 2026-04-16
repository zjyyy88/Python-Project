#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash batch_supercell_li_scan.sh [SOURCE_CIF]
#
# Optional env var:
#   SUPERCELL_BIN=/home/lilz/myworkflow/supercell/supercell

SOURCE_CIF="${1:-La2O3-C-3-zwbSEND-supercell.cif}"
SUPERCELL_BIN="${SUPERCELL_BIN:-/home/lilz/myworkflow/supercell/supercell}"
CIF_NAME="$(basename "$SOURCE_CIF")"

if [[ ! -f "$SOURCE_CIF" ]]; then
  echo "[ERROR] CIF template not found: $SOURCE_CIF" >&2
  exit 1
fi

if [[ ! -x "$SUPERCELL_BIN" ]]; then
  echo "[ERROR] supercell binary is not executable or not found: $SUPERCELL_BIN" >&2
  exit 1
fi

for i in $(seq 24 -1 0); do
  dir="Li${i}"
  mkdir -p "$dir"
  cp -f "$SOURCE_CIF" "$dir/$CIF_NAME"

  # Li occupancy follows folder name: Li24 -> 1.000000, Li0 -> 0.000000 (step 1/24).
  li_occ="$(awk -v n="$i" 'BEGIN { printf "%.6f", n / 24.0 }')"

  # La valence follows folder name: Li24 -> 2.25000, Li0 -> 3.00000 (step 0.03125).
  la_val="$(awk -v n="$i" 'BEGIN { printf "%.5f", 2.25 + (24 - n) * 0.03125 }')"

  cif_path="$dir/$CIF_NAME"
  LI_OCC="$li_occ" perl -i -pe '
    s/^(\s*Li\S*\s+\S+\s+\d+\s+\S+\s+\S+\s+\S+\s+)\S+/$1 . $ENV{LI_OCC}/e;
  ' "$cif_path"

  echo "[INFO] ${dir}: Li_occ=${li_occ}, La=${la_val}"

  (
    cd "$dir"
    echo "[RUN] $SUPERCELL_BIN -i $CIF_NAME -p Li*:c=+1 -p La*:c=${la_val} -p O*:c=-2 -q -c yes -m -n l10"
    "$SUPERCELL_BIN" \
      -i "$CIF_NAME" \
      -p "Li*:c=+1" \
      -p "La*:c=${la_val}" \
      -p "O*:c=-2" \
      -q -c yes -m -n l10
  )
done

echo "[DONE] All Li* folders processed successfully."
