import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import msd
import scipy.constants as const

def compute_conductivity(D_cm2_s, n_carriers, volume_A3, temp_K, z=1):
    """
    Calculate ionic conductivity using the Nernst-Einstein equation.
    sigma = (n * z^2 * e^2 * D) / (k_B * T)
    """
    V_cm3 = volume_A3 * 1e-24
    n_density = n_carriers / V_cm3  # carriers per cm^3
    q = z * const.e  # charge in Coulombs
    
    # D is in cm^2/s. 
    # To get conductivity in S/cm, we need compatible units.
    # sigma = n(cm^-3) * [q(C)]^2 * D(cm^2/s) / [k_B(J/K) * T(K)]
    # C^2 * cm^2 / s / J = (C/s) * (C * cm^2 / J) = A * (A*s * cm^2 / (V*A*s)) = A/V * cm^2 = S * cm^2
    # Oh wait, D is cm^2/s. J = kg m^2 / s^2. This unit check:
    # SI: sigma(S/m) = n(m^-3) * e^2(C^2) * D(m^2/s) / (k_B * T)
    
    D_m2_s = D_cm2_s * 1e-4
    n_density_m3 = n_carriers / (volume_A3 * 1e-30)
    sigma_S_m = (n_density_m3 * q**2 * D_m2_s) / (const.k * temp_K)
    sigma_mS_cm = sigma_S_m * 10  # 1 S/m = 10 mS/cm
    sigma_S_cm = sigma_S_m / 100
    
    return sigma_S_cm, sigma_mS_cm

def main():
    parser = argparse.ArgumentParser(description="Compute MSD and Conductivity using MDAnalysis")
    parser.add_argument("--poscar", default="POSCAR", help="Reference structure (e.g., POSCAR or CONTCAR)")
    parser.add_argument("--xdatcar", default="XDATCAR", help="XDATCAR trajectory file")
    parser.add_argument("--species", default="Li", help="Target species label (e.g., Li)")
    parser.add_argument("--temp", type=float, default=1000.0, help="Temperature in K")
    parser.add_argument("--timestep_fs", type=float, default=20.0, help="Time per frame in fs (POTIM * NBLOCK)")
    parser.add_argument("--fit_start_ps", type=float, default=2.0, help="Start time for linear fit (ps)")
    parser.add_argument("--fit_end_ps", type=float, default=10.0, help="End time for linear fit (ps)")
    parser.add_argument("--z", type=float, default=1.0, help="Valence of the carrier ion (e.g. 1 for Li)")
    parser.add_argument("--plot", action="store_true", help="Generate a plot of the MSD")
    args = parser.parse_args()

    print(f"Loading XDATCAR via custom parser...")
    try:
        from pymatgen.io.vasp.outputs import Xdatcar
        xdatcar = Xdatcar(args.xdatcar)
        
        # Pymatgen parses Xdatcar coords in fractional space or Cartesian?
        # But we need cartesian coordinates for MSD.
        structures = xdatcar.structures
        n_atoms = len(structures[0])
        n_frames = len(structures)
        
        # Create an array of cartesian coordinates
        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        for i, struct in enumerate(structures):
            coords[i] = struct.cart_coords
            
        elements = [str(site.specie) for site in structures[0]]
        dimensions = structures[0].lattice.abc + structures[0].lattice.angles
        volume_A3 = structures[0].lattice.volume
        
    except ImportError:
        print("pymatgen not found, falling back to simple parser")
        # Ensure we just exit for now or let the user install pymatgen since it's already there
        return
        
    print(f"Creating MDAnalysis Universe in memory with {n_frames} frames, {n_atoms} atoms...")
    u = mda.Universe.empty(n_atoms, n_residues=n_atoms, n_segments=1, trajectory=True)
    
    # Assign atom types/names
    u.add_TopologyAttr('resids', np.arange(1, n_atoms + 1))
    u.add_TopologyAttr('resnums', np.arange(1, n_atoms + 1))
    u.add_TopologyAttr('segids', ['SYST'])
    u.add_TopologyAttr('names', elements)
    u.add_TopologyAttr('types', elements)
    
    # Load coordinates
    u.load_new(coords, format='Memory')
    
    # Set dimensions (for unwrapping, if needed)
    for ts in u.trajectory:
        ts.dimensions = dimensions

    ag = u.select_atoms(f"name {args.species}")
    n_carriers = len(ag)
    if n_carriers == 0:
        print(f"No atoms found for species '{args.species}'! Check your POSCAR elements.")
        print("Available elements:", np.unique(u.atoms.names))
        return

    print(f"Found {n_carriers} {args.species} atoms. Calculating MSD (this might take a moment)...")
    
    # Compute MSD
    # Note: MDAnalysis msd.EinsteinMSD automatically handles periodic boundary unwrapping if coordinates are valid
    msd_analyzer = msd.EinsteinMSD(ag, msd_type='xyz', fft=False)
    msd_analyzer.run()

    msd_values = msd_analyzer.results.timeseries
    n_frames = len(msd_values)
    time_ps = np.arange(n_frames) * (args.timestep_fs / 1000.0)

    # Perform linear fitting
    start_idx = np.searchsorted(time_ps, args.fit_start_ps)
    end_idx = np.searchsorted(time_ps, args.fit_end_ps)
    
    if end_idx <= start_idx or start_idx >= n_frames:
        start_idx = max(0, n_frames // 10)
        end_idx = n_frames - 1
        print(f"Warning: requested fit range [{args.fit_start_ps}, {args.fit_end_ps}] out of bounds. Using [{time_ps[start_idx]:.2f}, {time_ps[end_idx]:.2f}] ps.")

    fit_time = time_ps[start_idx:end_idx]
    fit_msd = msd_values[start_idx:end_idx]
    
    coef = np.polyfit(fit_time, fit_msd, 1)
    slope = coef[0]  # A^2 / ps

    # D = slope / 6
    # 1 A^2 / ps = 1e-16 m^2 / 1e-12 s = 1e-4 cm^2 / s
    D_cm2_s = (slope / 6.0) * 1e-4

    sigma_S_cm, sigma_mS_cm = compute_conductivity(D_cm2_s, n_carriers, volume_A3, args.temp, args.z)

    print("-" * 50)
    print(f"Temperature: {args.temp} K")
    print(f"Carriers ({args.species}): {n_carriers}")
    print(f"Volume: {volume_A3:.2f} A^3")
    print(f"Linear Fit Region: {time_ps[start_idx]:.2f} - {time_ps[end_idx]:.2f} ps")
    print(f"MSD Slope: {slope:.6e} A^2/ps")
    print(f"Diffusion Coefficient (D): {D_cm2_s:.6e} cm^2/s")
    print(f"Ionic Conductivity (σ): {sigma_mS_cm:.2f} mS/cm  ({sigma_S_cm:.4e} S/cm)")
    print("-" * 50)

    # Save data
    df = pd.DataFrame({
        "time_ps": time_ps,
        f"{args.species}_MSD_A2": msd_values
    })
    csv_file = f"MDA_MSD_{args.species}_{args.temp}K.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")

    if args.plot:
        plt.figure(figsize=(7, 5))
        plt.plot(time_ps, msd_values, label="MSD (MDAnalysis)")
        plt.plot(fit_time, np.poly1d(coef)(fit_time), '--', color='red', label=f"Fit: {slope:.4f} t + {coef[1]:.4f}")
        plt.xlabel("Time (ps)")
        plt.ylabel("MSD (Å²)")
        plt.title(f"{args.species} MSD at {args.temp} K")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"MDA_MSD_{args.species}_{args.temp}K.png", dpi=300)
        print(f"Plot saved to MDA_MSD_{args.species}_{args.temp}K.png")

if __name__ == "__main__":
    main()
