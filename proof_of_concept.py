import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn import svm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from toolbox.plot_tools import plot_confusion_matrix

phd_data = 'C:/Users/ariel/Workspace/Organized PhD Catalog/'

sfh_catalog = Table.read(phd_data + 'catalog_PHO_flagged_lowz.fits')
galaxy_catalog = Table.read(phd_data + 'sample_flagged_lowz.fits')
models = Table.read(phd_data + 'cb17_7x16.fits')

sfh = sfh_catalog['SFH_m'].data.data[:,0:-1]

red_galaxies = galaxy_catalog['NUV_abs'] - galaxy_catalog['r_abs'] > 5
sfh_flag = np.array([np.any(np.isnan(galaxy_sfh)) for galaxy_sfh in sfh])

flag = sfh_flag | red_galaxies

upper_ages = np.unique(models['age_base_upp'])
lower_ages = np.unique(models['age_base'])
bin_len = np.array([(upper_ages[i]-lower_ages[i]) for i in range(16)])
mid_ages = np.array([lower_ages[i] + bin_len[i]/2 for i in range(16)])
young_ages = upper_ages < 0.5e9
old_ages = (upper_ages > 0.5e9) & (upper_ages < 3e9)

sfh = sfh[~flag]
sfh_catalog = sfh_catalog[~flag]
galaxy_catalog = galaxy_catalog[~flag]

sfh_young = np.array([galaxy_sfh[young_ages] for galaxy_sfh in sfh])
sfh_old = np.array([galaxy_sfh[old_ages] for galaxy_sfh in sfh])

sfh_young_mean = sfh_young.mean(axis=1)
sfh_young_sum = sfh_young.sum(axis=1)
sfh_young_std = sfh_young.std(axis=1)

sfh_old_mean = sfh_old.mean(axis=1)
sfh_old_sum = sfh_old.sum(axis=1)
sfh_old_std = sfh_old.std(axis=1)

fairness_factor = 1/6

enhanced = (sfh_young_sum / fairness_factor > 1.5 * sfh_old_sum)
subsided = (sfh_young_sum / fairness_factor < 0.5 * sfh_old_sum)

labels = np.array(['Enhanced' if enhanced[i] else 'Subsided' if subsided[i] else 'Other' for 
          i in range(len(sfh))])

plt.plot(np.log10(mid_ages), sfh[~enhanced & ~subsided].mean(axis=0))
plt.plot(np.log10(mid_ages), sfh[subsided].mean(axis=0))
plt.plot(np.log10(mid_ages), sfh[enhanced].mean(axis=0))
plt.plot(np.log10(mid_ages), sfh.mean(axis=0), '--k')
plt.xlim(8, 9.75)

classifier = svm.SVC()

classifier.fit(sfh[10000:-1], labels[10000:-1])
predicted_labels = classifier.predict(sfh[0:10000])

plot_confusion_matrix(confusion_matrix(labels[0:10000], predicted_labels[0:10000]),
                      classes=['Enhanced', 'Subsided', 'Other'], normalize=True)