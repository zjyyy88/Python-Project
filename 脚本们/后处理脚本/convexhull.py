from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

with MPRester("HiDNgsh0Zz5RnMjLIjH3HxfFjTwnRIgb") as mpr:

    # Obtain only corrected GGA and GGA+U ComputedStructureEntry objects
    entries = mpr.get_entries_in_chemsys(elements=["Li", "Fe", "O"], 
                                         additional_criteria={"thermo_types": ["GGA_GGA+U"]}) 
    # Construct phase diagram
    pd = PhaseDiagram(entries)
    
    # Plot phase diagram
    PDPlotter(pd).get_plot()
