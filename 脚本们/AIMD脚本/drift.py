#!/usr/bin/env python3
# unwrap_drift.py  : 去掉 XDATCAR 质心漂移
import numpy as np
import sys

fname = sys.argv[1] if len(sys.argv)>1 else 'XDATCAR'
lines = open(fname).readlines()

idx = 0
header = []
while not lines[idx].strip().startswith('Direct'):
    header.append(lines[idx]); idx+=1
header.append(lines[idx]); idx+=1   # 保留 "Direct configuration" 行

natoms = sum(map(int, header[6].split()))
nframes = (len(lines)-idx) // (natoms+1)

with open('XDATCAR_nodrift','w') as f:
    f.write(''.join(header))
    for k in range(nframes):
        f.write(header[-1])          # "Direct configuration=..."
        coords = np.loadtxt(lines[idx+1:idx+1+natoms], dtype=float)
        com = coords.mean(axis=0)    # 质心
        coords -= com                # 去漂移
        np.savetxt(f, coords, fmt='%20.16f')
        idx += natoms+1
print('written to XDATCAR_nodrift')
