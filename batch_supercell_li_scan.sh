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

  # Li2 occupancy: Li24->Li8 is 1.0000 -> 0.0000 with step 0.0625.
  # Li1 occupancy: Li8->Li0 is 1.0000 -> 0.0000 with step 0.1250.
  if (( i >= 8 )); then
    li2_occ="$(awk -v n="$i" 'BEGIN { printf "%.4f", (n - 8) * 0.0625 }')"
    li1_occ="1.0000"
  else
    li2_occ="0.0000"
    li1_occ="$(awk -v n="$i" 'BEGIN { printf "%.4f", n * 0.1250 }')"
  fi

  # La valence: Li24->Li0 is 2.25000 -> 3.00000 with step 0.03125.
  la_val="$(awk -v n="$i" 'BEGIN { printf "%.5f", 2.25 + (24 - n) * 0.03125 }')"

  cif_path="$dir/$CIF_NAME"
  LI1_OCC="$li1_occ" LI2_OCC="$li2_occ" perl -i -pe '
    s/^(\s*Li1\s+)\S+/$1 . $ENV{LI1_OCC}/e;
    s/^(\s*Li2\s+)\S+/$1 . $ENV{LI2_OCC}/e;
  ' "$cif_path"

  echo "[INFO] ${dir}: Li1=${li1_occ}, Li2=${li2_occ}, La=${la_val}"

  (
    cd "$dir"
    "$SUPERCELL_BIN" \
      -i "$CIF_NAME" \
      -p "Li*:c=+1" \
      -p "La*:c=${la_val}" \
      -p "O*:c=-2" \
      -q -c yes -m -n l10
  )
done

echo "[DONE] All Li* folders processed successfully."