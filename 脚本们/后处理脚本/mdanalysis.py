import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from MDAnalysis.tests.datafiles import RANDOM_WALK, RANDOM_WALK_TOPO
import numpy as np
u = mda.Universe(RANDOM_WALK, RANDOM_WALK_TOPO)
MSD = msd.EinsteinMSD(u, select='all', msd_type='xyz', fft=True)
MSD.run()
msd =  MSD.results.timeseries
import matplotlib.pyplot as plt
nframes = MSD.n_frames
timestep = 2 # this needs to be the actual time between frames
lagtimes = np.arange(nframes)*timestep # make the lag-time axis
fig = plt.figure()
ax = plt.axes()
# plot the actual MSD
ax.plot(lagtimes, msd, lc="black", ls="-", label=r'3D random walk')
exact = lagtimes*6
# plot the exact result
ax.plot(lagtimes, exact, lc="black", ls="--", label=r'$y=2 D\tau$')
plt.show()
plt.loglog(lagtimes, msd)
plt.show()
from scipy.stats import linregress
start_time = 20
start_index = int(start_time/timestep)
end_time = 60
linear_model = linregress(lagtimes[start_index:end_index],
                                              msd[start_index:end_index])
slope = linear_model.slope
error = linear_model.rvalue
# dim_fac is 3 as we computed a 3D msd with 'xyz'
D = slope * 1/(2*MSD.dim_fac)