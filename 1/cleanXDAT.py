#!/usr/bin/env python3
import argparse
from pathlib import Path


def _next_nonempty(lines, idx):
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise ValueError("XDATCAR header is incomplete.")
    line = lines[idx].rstrip("\n")
    return line, idx + 1


def parse_header(lines):
    idx = 0
    header = []

    line, idx = _next_nonempty(lines, idx)
    header.append(line)  # comment
    line, idx = _next_nonempty(lines, idx)
    header.append(line)  # scale

    for _ in range(3):  # lattice vectors
        line, idx = _next_nonempty(lines, idx)
        header.append(line)

    names_line, idx = _next_nonempty(lines, idx)
    counts_line, idx = _next_nonempty(lines, idx)
    header.append(names_line)
    header.append(counts_line)

    names = names_line.split()
    counts = [int(x) for x in counts_line.split()]
    if len(names) != len(counts):
        raise ValueError("Element names/counts mismatch in header.")

    n_atoms = sum(counts)
    return header, n_atoms, idx


def parse_coord(line):
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def collect_frames(lines, start_idx, n_atoms):
    frames = []
    i = start_idx
    while i < len(lines):
        if "Direct configuration" not in lines[i]:
            i += 1
            continue

        i += 1
        coords = []
        found_next = False

        while i < len(lines) and len(coords) < n_atoms:
            if "Direct configuration" in lines[i]:
                found_next = True
                break
            coord = parse_coord(lines[i])
            if coord is not None:
                coords.append(coord)
            i += 1

        if len(coords) == n_atoms:
            frames.append(coords)

        if found_next and len(coords) == 0:
            i += 1

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Clean XDATCAR to a single-header + Direct configuration blocks format."
    )
    parser.add_argument("xdatcar", help="Input XDATCAR path")
    parser.add_argument("-o", "--out", default="XDATCAR_clean", help="Output XDATCAR path")
    args = parser.parse_args()

    in_path = Path(args.xdatcar)
    lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header, n_atoms, idx = parse_header(lines)
    frames = collect_frames(lines, idx, n_atoms)
    if not frames:
        raise ValueError("No Direct configuration blocks found.")

    out_lines = list(header)
    for k, coords in enumerate(frames, start=1):
        out_lines.append(f"Direct configuration= {k:5d}")
        for x, y, z in coords:
            out_lines.append(f"  {x: .8f}  {y: .8f}  {z: .8f}")

    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} with {len(frames)} frames, {n_atoms} atoms per frame.")


if __name__ == "__main__":
    main()
