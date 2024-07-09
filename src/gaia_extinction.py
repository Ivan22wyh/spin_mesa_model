# %%
from glob import glob
import myfunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import cluster
from tqdm import tqdm
from astropy.table import Table
from dustapprox.extinction import CCM89, F99
from dustapprox.io import svo
from scipy.interpolate import interpn

curves = [CCM89(), F99()]
Rv = 3.1

models = glob('models/Kurucz2003all/*.fl.dat.txt')
apfields = ['teff', 'logg', 'feh', 'alpha']

which_filters = ['GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp', 'GAIA/GAIA3.G']
passbands = svo.get_svo_passbands(which_filters)

for pb in passbands:
   plt.plot(pb.wavelength.to('nm'), pb.transmit, label=pb.name)

plt.legend(loc='upper right', frameon=False)

plt.xlabel('wavelength [nm]')
plt.ylabel('transmission')
plt.tight_layout()
plt.show()


Av = np.arange(0, 10.01, 0.2)

logs = []
for fname in tqdm(models):
    data = svo.spectra_file_reader(fname)
    # extract model relevant information
    lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
    lamb = data['data']['WAVELENGTH'].values * lamb_unit
    flux = data['data']['FLUX'].values * flux_unit
    apvalues = [data[k]['value'] for k in apfields]

    # wavelength definition varies between models
    alambda_per_av = curves[0](lamb, Av=1.0, Rv=Rv)

    # Dust magnitudes
    columns = apfields + ['passband', 'mag0', 'mag', 'A0', 'Ax']
    for pk in passbands:
        mag0 = -2.5 * np.log10(pk.get_flux(lamb, flux).value)
        # we redo av = 0, but it's cheap, allows us to use the same code
        for av_val in Av:
            new_flux = flux * np.exp(- alambda_per_av * av_val)
            mag = -2.5 * np.log10(pk.get_flux(lamb, new_flux).value)
            delta = (mag - mag0)
            logs.append(apvalues + [pk.name, mag0, mag, av_val, delta])

logs = pd.DataFrame.from_records(logs, columns=columns)


# %%




# %%
err = []

def get_val(df, teff, logg, av, feh, passband):
    result = df[(df['teff'] == teff) & 
                (df['logg'] == logg) & 
                (df['A0'] == av) & 
                (df['feh'] == feh) &
                (df['passband'] == passband)
                ]

    # 如果 result 不为空，表示找到了满足条件的行
    if not result.empty:
        return np.mean(result['Ax'])
    else:
        err.append([teff, logg, av, feh])
    
get_val(logs, 6000, 2, 0.2, -0.5, 'GAIA_GAIA3.Gbp')



# %%


# %%
teff_grid = [t for t, c in myfunc.count_element(logs['teff'])]
logg_grid = [g for g, c in myfunc.count_element(logs['logg'])]
av_grid = [av for av, c in myfunc.count_element(logs['A0'])]
feh_grid = [feh for feh, c in myfunc.count_element(logs['feh'])]

# %%
values = np.zeros((len(teff_grid), len(logg_grid), len(av_grid), len(feh_grid)))
shape = values.shape

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in tqdm(range(shape[2])):
            for s in range(shape[3]):
                values[i, j, k, s] = get_val(logs, 
                                             teff=teff_grid[i],
                                             logg=logg_grid[j],
                                             av=av_grid[k],
                                             feh=feh_grid[s],
                                             passband='GAIA_GAIA3.Gbp'
                                             )

# %%
points = (teff_grid, logg_grid, av_grid, feh_grid)
xi = [9000, 1.3, 0.2, 0.2]

res = interpn(
    points,
    values, xi,
    # 外插时自动填充值，而不是nan
    fill_value=True,
    # 允许外插
    bounds_error=False,
    method='linear')
print(res[0])

# %%


# %%

# %%

# %%
gaia_extinction = Table.read("resources/f99.fits")
gaia_extinction_bp = gaia_extinction[np.where(gaia_extinction['passband']=='GAIA_GAIA3.Gbp')[0]]
gaia_extinction_rp = gaia_extinction[np.where(gaia_extinction['passband']=='GAIA_GAIA3.Grp')[0]]

# %%
teff_grid = np.sort([f for f, c in myfunc.count_element(gaia_extinction['teff'])])
a0_grid = np.sort([a for a, c in myfunc.count_element(gaia_extinction['A0'])])

bp_extinction_grid = np.zeros((len(teff_grid), len(a0_grid)))
shape = bp_extinction_grid.shape

for i in tqdm(range(shape[0])):
    for j in range(shape[1]):
        index1 = np.where(gaia_extinction_bp['teff']==teff_grid[i])[0]
        res = gaia_extinction_bp[index1]
        index2 = np.where(res['A0']==a0_grid[j])[0]
        res = res[index2]
        bp_extinction_grid[i,j] = np.mean(res['Ax'])

rp_extinction_grid = np.zeros((len(teff_grid), len(a0_grid)))

for i in tqdm(range(shape[0])):
    for j in range(shape[1]):
        index1 = np.where(gaia_extinction_rp['teff']==teff_grid[i])[0]
        res = gaia_extinction_rp[index1]
        index2 = np.where(res['A0']==a0_grid[j])[0]
        res = res[index2]
        rp_extinction_grid[i,j] = np.mean(res['Ax'])

points = (teff_grid, a0_grid)

np.savez("resources/gaia_extinction_grid.npz", bp=bp_extinction_grid, rp=rp_extinction_grid)


# %%
cluster = Table.read("resources/open_cluster_member_v6.fits")

# %%
param = [[s['Teff'], s['Av']] for s in cluster]

# %%
bp_extinction, rp_extinction = [], []

for x in tqdm(param):
    try:
        res = interpn(
            points,
            bp_extinction_grid, x,
            # 外插时自动填充值，而不是nan
            fill_value=True,
            # 允许外插
            bounds_error=False,
            method='cubic')
        bp_extinction.append(res)
    except ValueError:
        bp_extinction.append([np.nan])

for x in tqdm(param):
    try:
        res = interpn(
            points,
            rp_extinction_grid, x,
            # 外插时自动填充值，而不是nan
            fill_value=True,
            # 允许外插
            bounds_error=False,
            method='cubic')
        rp_extinction.append(res)
    except ValueError:
        rp_extinction.append([np.nan])


# %%
cluster['Av_bp'] = [x[0] for x in bp_extinction]
cluster['Av_rp'] = [x[0] for x in rp_extinction]

cluster.write("resources/open_cluster_member_v6.fits", overwrite=True)



# %%


# %%







# %%
