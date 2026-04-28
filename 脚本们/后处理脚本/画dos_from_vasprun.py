from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.io.vasp import Vasprun


# Fixed input file requested by user.
VASPRUN_PATH = Path(
    "D:\\aaazjy\\2025.4.1\\InCl3\u00b73H2O\\case\\reaction\\newVCl2OH-27712-USPEX-InCl2OH\\vasprun.xml"
)
OUTDIR = VASPRUN_PATH.parent / "dos_outputs"
XMIN, XMAX = -8, 8
SIGMA = 0.05

# Selected element DOS, e.g. ["V", "Cl"]. Empty list means skip this section.
#TARGET_ELEMENTS = ["In","O" ,"H","Cl"]
TARGET_ELEMENTS = []
# Selected atom indices for local DOS. Set INDEX_BASE to 1 if you use VESTA-style index.
TARGET_SITE_INDICES = [1, 3,4,5,6,8,10]
INDEX_BASE = 1

# Optional: also output per-site s/p/d decomposed DOS.
PLOT_SITE_SPD = True


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def main() -> None:
    if not VASPRUN_PATH.exists():
        raise FileNotFoundError(f"vasprun.xml not found: {VASPRUN_PATH}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    vrun = Vasprun(str(VASPRUN_PATH), parse_potcar_file=False)

    # Two core objects.
    dos_data = vrun.tdos
    bs_data = None
    try:
        bs_data = vrun.get_band_structure()
    except Exception as exc:
        print(f"[Warning] Could not parse band structure: {exc}")

    # 1) Total DOS
    tdos_plotter = DosPlotter(sigma=SIGMA, zero_at_efermi=True)
    tdos_plotter.add_dos("Total DOS", dos_data)
    ax_tdos = tdos_plotter.get_plot(xlim=[XMIN, XMAX])
    ax_tdos.axvline(0, color="k", linestyle="--", linewidth=1)
    ax_tdos.figure.tight_layout()
    ax_tdos.figure.savefig(OUTDIR / "tdos.png", dpi=300)
    plt.close(ax_tdos.figure)

    # 2) Element DOS
    complete_dos = vrun.complete_dos
    structure = complete_dos.structure
    element_dos = complete_dos.get_element_dos()

    ele_plotter = DosPlotter(sigma=SIGMA, zero_at_efermi=True)
    ele_plotter.add_dos_dict(element_dos)
    ax_ele = ele_plotter.get_plot(xlim=[XMIN, XMAX])
    ax_ele.axvline(0, color="k", linestyle="--", linewidth=1)
    ax_ele.figure.tight_layout()
    ax_ele.figure.savefig(OUTDIR / "element_dos.png", dpi=300)
    plt.close(ax_ele.figure)

    # 2.1) Selected element DOS
    if TARGET_ELEMENTS:
        wanted = {sym.strip().upper() for sym in TARGET_ELEMENTS if sym.strip()}
        selected_element_dos = {
            str(ele): dos
            for ele, dos in element_dos.items()
            if str(ele).upper() in wanted
        }
        if selected_element_dos:
            sel_plotter = DosPlotter(sigma=SIGMA, zero_at_efermi=True)
            sel_plotter.add_dos_dict(selected_element_dos)
            ax_sel = sel_plotter.get_plot(xlim=[XMIN, XMAX])
            ax_sel.axvline(0, color="k", linestyle="--", linewidth=1)
            ax_sel.figure.tight_layout()
            ax_sel.figure.savefig(OUTDIR / "selected_element_dos.png", dpi=300)
            plt.close(ax_sel.figure)
        else:
            available = ", ".join(sorted(str(ele) for ele in element_dos))
            print(
                "[Warning] No matching elements found for TARGET_ELEMENTS. "
                f"Available elements: {available}"
            )

    # 2.2) Selected site local DOS (LDOS)
    if TARGET_SITE_INDICES:
        n_sites = len(structure)
        index_header = [
            "# Site index map",
            f"# INDEX_BASE = {INDEX_BASE}",
            f"# Total sites = {n_sites}",
            "index\tspecies\tfrac_coords",
        ]
        index_lines = []
        for i, site in enumerate(structure):
            idx_show = i + 1 if INDEX_BASE == 1 else i
            frac = ", ".join(f"{v:.6f}" for v in site.frac_coords)
            index_lines.append(f"{idx_show}\t{site.species_string}\t{frac}")
        (OUTDIR / "site_index_map.txt").write_text(
            "\n".join(index_header + index_lines) + "\n", encoding="utf-8"
        )

        for idx_user in TARGET_SITE_INDICES:
            idx0 = idx_user - 1 if INDEX_BASE == 1 else idx_user
            if idx0 < 0 or idx0 >= n_sites:
                print(
                    f"[Warning] Site index {idx_user} is out of range "
                    f"for n_sites={n_sites} (INDEX_BASE={INDEX_BASE})."
                )
                continue

            site = structure[idx0]
            site_label = f"site_{idx_user}_{site.species_string}"
            site_safe = _safe_name(site_label)

            site_dos = complete_dos.get_site_dos(site)
            site_plotter = DosPlotter(sigma=SIGMA, zero_at_efermi=True)
            site_plotter.add_dos(site_label, site_dos)
            ax_site = site_plotter.get_plot(xlim=[XMIN, XMAX])
            ax_site.axvline(0, color="k", linestyle="--", linewidth=1)
            ax_site.figure.tight_layout()
            ax_site.figure.savefig(OUTDIR / f"{site_safe}_ldos.png", dpi=300)
            plt.close(ax_site.figure)

            if PLOT_SITE_SPD:
                spd_dos = complete_dos.get_site_spd_dos(site)
                if spd_dos:
                    spd_plotter = DosPlotter(sigma=SIGMA, zero_at_efermi=True)
                    spd_plotter.add_dos_dict({str(k): v for k, v in spd_dos.items()})
                    ax_spd = spd_plotter.get_plot(xlim=[XMIN, XMAX])
                    ax_spd.axvline(0, color="k", linestyle="--", linewidth=1)
                    ax_spd.figure.tight_layout()
                    ax_spd.figure.savefig(OUTDIR / f"{site_safe}_spd.png", dpi=300)
                    plt.close(ax_spd.figure)

    # 3) Optional band summary from bs_data
    if bs_data is not None:
        gap = bs_data.get_band_gap()
        vbm = bs_data.get_vbm().get("energy", None)
        cbm = bs_data.get_cbm().get("energy", None)
        summary = [
            f"band_gap(eV): {gap.get('energy', None)}",
            f"direct_gap: {gap.get('direct', None)}",
            f"transition: {gap.get('transition', None)}",
            f"VBM(eV): {vbm}",
            f"CBM(eV): {cbm}",
        ]
        (OUTDIR / "band_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Done. Results are in: {OUTDIR}")


if __name__ == "__main__":
    main()
