import numpy as np
import pandas as pd
from astropy.table import Table
from starlight_toolkit.synphot import synmag

phd_data = 'C:/Users/ariel/Workspace/Organized PhD Catalog/'
models_dir = 'C:/Users/ariel/Workspace/GASP/SFH Classification/Data/models/stellar+nebular/'
filters_dir = 'C:/Users/ariel/Workspace/GASP/SFH Classification/Data/filters/'

filter_files = [filters_dir + 'HST_WFC3_UVIS2.F275W.dat',
                filters_dir + 'HST_WFC3_UVIS2.F336W.dat',
                filters_dir + 'HST_WFC3_UVIS2.F606W.dat',
                filters_dir + 'HST_WFC3_UVIS2.F680N.dat',
                filters_dir + 'HST_WFC3_UVIS2.F814W.dat']

starlight_catalog = Table.read(phd_data + 'catalog_PHO_flagged_lowz.fits')
galaxy_catalog = Table.read(phd_data + 'sample_flagged_lowz.fits')

simulated_mags = []

z = 0.044631

for file in starlight_catalog['file']:

    file_index = np.argwhere(starlight_catalog['file']==file)[0][0]

    print(file, file_index)

    try:
        wl, model, model_stellar, nebular_model = np.genfromtxt(models_dir + file + '.mod').transpose()
    except:
        simulated_mags.append([np.nan for i in range(5)])
        continue

    model *= 1e-17 / (1+z)
    wl *= (1+z)

    simulated_mags.append([synmag(wl, model, hst_filter) for hst_filter in filter_files])


simulated_mags = np.array(simulated_mags)

simulation_catalog = Table()
simulation_catalog['file'] = starlight_catalog['file']
simulation_catalog['hst_magnitudes'] = simulated_mags
simulation_catalog['sfh_m'] = starlight_catalog['SFH_m']

simulation_catalog.write('C:/Users/ariel/Workspace/GASP/SFH Classification/Data/simulated_catalog.fits', overwrite=True)