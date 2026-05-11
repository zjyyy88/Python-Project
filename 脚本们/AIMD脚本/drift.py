#!/usr/bin/env python3
# unwrap_drift.py  : 去掉 XDATCAR 质心漂移
import numpy as np
import sys

fname = sys.argv[1] if len(sys.argv) > 1 else "XDATCAR"
lines = open(fname).readlines()


def _parse_counts_line(line):
    parts = line.split()
    if not parts:
        return None
    for p in parts:
        if not p.isdigit():
            return None
    return [int(p) for p in parts]


def _find_counts_line(header):
    for line in reversed(header):
        counts = _parse_counts_line(line)
        if counts is not None:
            return counts
    raise ValueError("Cannot locate atom counts line in XDATCAR header.")

idx = 0
header = []
while idx < len(lines) and not lines[idx].strip().startswith(("Direct", "Cartesian")):
    header.append(lines[idx])
    idx += 1
if idx >= len(lines):
    raise ValueError("XDATCAR does not contain 'Direct configuration' lines.")

counts = _find_counts_line(header)
natoms = sum(counts)

with open("XDATCAR_nodrift", "w") as f:
    f.write("".join(header))
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith(("Direct", "Cartesian")):
            idx += 1
            continue
        direct_line = lines[idx]
        idx += 1
        coords = []
        while len(coords) < natoms and idx < len(lines):
            raw = lines[idx].strip()
            idx += 1
            if not raw:
                continue
            if raw.startswith(("Direct", "Cartesian")):
                # Skip unexpected frame markers; continue searching for coords.
                continue
            parts = raw.split()
            if len(parts) < 3:
                continue
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
        if len(coords) < natoms:
            raise ValueError("Not enough coordinate lines before EOF.")
        coords = np.array(coords, dtype=float)
        com = coords.mean(axis=0)
        coords -= com
        f.write(direct_line)
        np.savetxt(f, coords, fmt="%20.16f")
print('written to XDATCAR_nodrift')
