# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery
import astropy.units as u

cluster = Table.read('/mnt/d/Spin/resources/open_cluster_member_v4.fits', format='fits')

def exclude_zero(arr, zero, padding=True):
    if padding:
        arr[zero] = 0.1
        return arr     
    return np.array([x for i,x in enumerate(arr) if i not in zero])

# %%
ra, dec = np.array(cluster['RAdeg']), np.array(cluster['DEdeg'])
#plx = np.array(cluster['Plx_x'])
#distance = np.abs(np.array([1/(x/1000) for x in cluster['Plx_x']]))
distance = np.abs(np.array(cluster['rpgeo']))

#%%
""" 
zero = np.where(distance<0)[0]

distance = exclude_zero(distance, zero)
 """
# %%
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=distance*u.pc, frame='icrs')


# %%
bayestar = BayestarQuery(max_samples=1, version='bayestar2019')

ebv = [bayestar(coords[i], mode='best') for i in range(len(coords))]
Av = 2.742*np.array(ebv)

np.savez("/mnt/d/Spin/a.npz", ebv=ebv, Av=Av)

# %%

cluster['ebv'] = ebv
cluster['Av'] = Av

cluster.write('/mnt/d/Spin/resources/open_cluster_member_v4.fits', overwrite=True)




# %%
Av = np.load("/mnt/d/Spin/a.npz")['Av']
ebv = np.load("/mnt/d/Spin/a.npz")['ebv']
# %%
len(Av)
# %%
ra,dec, distance = 153.3715336312, -54.66429728493, 461.277771
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=distance*u.pc, frame='icrs')
bayestar = BayestarQuery(max_samples=1, version='bayestar2019')
bayestar(coords, mode='best')