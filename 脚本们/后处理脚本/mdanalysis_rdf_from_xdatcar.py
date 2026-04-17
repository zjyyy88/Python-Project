import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.coordinates.memory import MemoryReader
from pymatgen.io.vasp import Xdatcar


def load_universe_from_xdatcar(xdatcar_path: Path, start: int, stop: int | None, stride: int):
    xd = Xdatcar(str(xdatcar_path))
    structures = xd.structures[start:stop:stride]
    if not structures:
        raise ValueError("No frames selected. Check start/stop/stride.")

    n_atoms = len(structures[0])
    coords = np.asarray([s.cart_coords for s in structures], dtype=np.float32)
    dims = np.asarray(
        [
            [s.lattice.a, s.lattice.b, s.lattice.c, s.lattice.alpha, s.lattice.beta, s.lattice.gamma]
            for s in structures
        ],
        dtype=np.float32,
    )
    names = [site.specie.symbol for site in structures[0]]

    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", names)
    u.load_new(coords, format=MemoryReader, order="fac", dimensions=dims)

    return u, sorted(set(names))


def compute_rdf(u, ref_elem: str, target_elem: str, r_min: float, r_max: float, n_bins: int):
    ag1 = u.select_atoms(f"name {ref_elem}")
    ag2 = u.select_atoms(f"name {target_elem}")

    if ag1.n_atoms == 0:
        raise ValueError(f"Reference group empty: {ref_elem}")
    if ag2.n_atoms == 0:
        raise ValueError(f"Target group empty: {target_elem}")

    rdf = InterRDF(
        ag1,
        ag2,
        nbins=n_bins,
        range=(r_min, r_max),
        exclusion_block=None,
        norm="rdf",
    )
    rdf.run()

    return rdf.results.bins, rdf.results.rdf, ag1.n_atoms, ag2.n_atoms


def main():
    parser = argparse.ArgumentParser(description="Compute RDF from VASP XDATCAR using MDAnalysis.")
    parser.add_argument("--xdatcar", required=True, help="Path to XDATCAR file")
    parser.add_argument("--ref", default="Li", help="Reference element (default: Li)")
    parser.add_argument("--target", default="Cl", help="Target element (default: Cl)")
    parser.add_argument("--rmin", type=float, default=0.0, help="RDF range minimum in Angstrom")
    parser.add_argument("--rmax", type=float, default=8.0, help="RDF range maximum in Angstrom")
    parser.add_argument("--bins", type=int, default=160, help="Number of RDF bins")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--out", default="rdf", help="Output prefix")
    args = parser.parse_args()

    xdatcar_path = Path(args.xdatcar)
    if not xdatcar_path.exists():
        raise FileNotFoundError(f"XDATCAR file not found: {xdatcar_path}")

    u, available_elements = load_universe_from_xdatcar(
        xdatcar_path=xdatcar_path,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )

    bins, rdf_vals, n_ref, n_target = compute_rdf(
        u,
        ref_elem=args.ref,
        target_elem=args.target,
        r_min=args.rmin,
        r_max=args.rmax,
        n_bins=args.bins,
    )

    out_csv = Path(f"{args.out}_{args.ref}_{args.target}.csv")
    out_png = Path(f"{args.out}_{args.ref}_{args.target}.png")

    df = pd.DataFrame({"r_A": bins, "g_r": rdf_vals})
    df.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    plt.plot(bins, rdf_vals, lw=2)
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title(f"RDF: {args.ref}-{args.target}")
    plt.xlim(args.rmin, args.rmax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print("RDF finished.")
    print(f"Frames used: {len(u.trajectory)}")
    print(f"Available elements: {available_elements}")
    print(f"Selection: {args.ref} ({n_ref}) -> {args.target} ({n_target})")
    print(f"CSV: {out_csv.resolve()}")
    print(f"PNG: {out_png.resolve()}")


if __name__ == "__main__":
    main()
#c:/Users/ZHANGJY02/PycharmProjects/PythonProject/.venv/Scripts/python.exe c:/Users/ZHANGJY02/PycharmProjects/PythonProject/脚本们/后处理脚本/mdanalysis_rdf_from_xdatcar.py
#  --xdatcar "D:\aaazjy\zjyyyyy\halide water adsorption\zjy-calc\Bi-dopingLYC\QJHCONTCAR-Bi0.33\MSD\XDATCAR-1000K" --ref Li --target Cl --rmin 0 --rmax 8 --bins 160 --stride 1 --out rdf_1000K