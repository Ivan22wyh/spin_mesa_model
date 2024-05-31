# %%
import os
import shutil
import numpy as np
import myfunc
import matplotlib.pyplot as plt
import warnings
from scipy import cluster
from tqdm import tqdm
from astropy.table import Table
from scipy.signal import find_peaks
from lightkurve import search_lightcurve, LightCurve, RegressionCorrector, DesignMatrix, DesignMatrixCollection
from astropy.timeseries import LombScargle


warnings.filterwarnings('ignore')

def equalize_averages(seq1, seq2):

    avg_seq1 = sum(seq1) / len(seq1)
    avg_seq2 = sum(seq2) / len(seq2)

    difference = avg_seq1 - avg_seq2
    adjusted_seq2 = [x + difference for x in seq2]

    return adjusted_seq2


def find_duplicate_paths(paths):
    path_dict = {}
    for i, path in enumerate(paths):
        if paths.count(path) > 1:
            if path in path_dict:
                path_dict[path].append(i)
            else:
                path_dict[path] = [i]
    return path_dict

def copy_files(paths):
    # 获取要复制到的路径
    dest_path = paths[0]

    # 遍历每个源路径
    for src_path in paths[1:]:
        # 遍历每个源路径下的所有文件
        for root, dirs, files in os.walk(src_path):
            for file in files:
                # 构建源文件和目标文件的路径
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)

                # 复制文件
                shutil.copy(src_file, dest_file)
    
    

# %%
x = np.load("f.npz")['emsto']
y = np.load("f.npz")['non']
c = "NGC_2548"

try:
    period = np.load(f"{c}.npz".lower().replace('_', ''))['p']
    dquality = np.load(f"{c}.npz".lower().replace('_', ''))['dq']

except:
    period = np.zeros((1000))
    dquality = np.zeros((1000))

dr3 = np.array(['0' for _ in range(1000)])

# %%

# %%  -----------------------------------PATHOS---------------------------------------
# 指定路径
folder_path = 'D:/Spin/mastDownload/HLSP'

# 获取文件夹列表
folder_list = myfunc.list_folders(folder_path)
fits_file_list = np.array(myfunc.list_files(folder_path, 'fits'))
#fits_file_list = file_list[myfunc.file_extention(file_list, pattern="fits")]

# %% Intersect the tess catalog and pathos 
tess = Table.read('resources/cluster_pro_v8.fits', format='fits')
tess = tess[np.where(tess['Cluster']==c)[0]]
tess = tess[np.where(tess['isbinary'] == 0)[0]]
tess = tess[np.where(tess['Proba'] > 0.5)[0]]
cluster = Table.read("resources/cluster_pro_v8.fits")
emsto = np.load("resources/extended_v2.npz")['emsto']

pathos_list = np.array([int(x.split('tic-')[1].split('-s')[0])
                       for x in folder_list], dtype=int)
tess_result = tess['TIC']
tess_result_vsini = [x['TIC'] for x in tess 
                     if (x['vsini_ph'] != -9999) or (x['vsini_kurucz'] != -9999) or (x['vsini_elodie'] != -9999)]

target = myfunc.intersection(tess_result, pathos_list)
target_vsini = myfunc.intersection(tess_result_vsini, pathos_list)

pathos_target_list = np.array([int(x.split('tic-')[1].split('-s')[0])
                       for x in folder_list[target_vsini[2]]], dtype=int)

fits_file_tic_list = np.array([int(x.split('tic-')[1].split('-s')[0])
                       for x in fits_file_list], dtype=int)

fits_file_target = fits_file_list[myfunc.intersection(pathos_target_list, fits_file_tic_list)[2]]

len(fits_file_target)

# %%
plt.rcParams.update({'font.size': 14}) 
def get_flux(lc, tmag):
    thres = 3

    time = lc['TIME']
    if tmag < 7:
        flux = lc['AP4_FLUX_COR']
    elif (tmag > 7) and (tmag < 9):
        flux = lc['AP3_FLUX_COR']
    elif (tmag > 9) and (tmag < 10.5):
        flux = lc['AP2_FLUX_COR']    
    elif (tmag > 10.5) and (tmag < 13.5):
        flux = lc['PSF_FLUX_COR']
    elif tmag > 13.5:
        flux = lc['AP1_FLUX_COR']

    clipped_lc = LightCurve(time=np.array(time, dtype=float), flux=np.array(flux, dtype=float))
    clipped_lc = clipped_lc.remove_nans().remove_outliers(sigma=thres)
    
    dm = DesignMatrix(clipped_lc.time.value, name='time').pca(5)
    dm_collection = DesignMatrixCollection([dm])

    corrector = RegressionCorrector(clipped_lc)
    corrected_lc = corrector.correct(dm_collection)

    #time, flux = np.array(corrected_lc.time.value), np.array(corrected_lc.flux.value)-1
    time, flux = np.array(clipped_lc.time.value), np.array(clipped_lc.flux.value)-1

    #time, flux = time[np.where(flux>10)[0]], flux[np.where(flux>10)[0]]

    return time, flux

def period_pathos(time, flux):

    figsize = (18, 8)
    stop = 1842
    res = []

    #plt.scatter(time,flux)

    fig, ax = plt.subplots(3,3, figsize=figsize)
    cutoff = (np.where(time > 1503)[0][0], np.where(time > 1504)[0][0], np.where(time > 1314)[0][0])

    time1, flux1 = np.array(time[:cutoff[0]]), np.array(flux[:cutoff[0]] )
    time2, flux2 = np.array(time[cutoff[1]:]), np.array(flux[cutoff[1]:])
    #flux2 = equalize_averages(flux1, flux2)
    #time3, flux3 = time[cutoff[3]:], flux[cutoff[3]:]

    for i in range(3):

        if i == 0: x, y = time1, flux1 
        elif i == 1: x, y = time2, flux2
        #elif i == 2: x, y = time3, flux3
        #elif i == 2: x, y = np.concatenate((time1, time2)), np.concatenate((flux1, flux2))

        if i == 2: x,y = time, flux

        ax[0][i].scatter(x, y, s=3, c='r')
        #ax[0][i].set_title("PATHOS Normalized Lightcurve")

        acf = myfunc.acf(y, smooth=True, sigma=5)
        tao = np.array([(time[2] - time[1])*k for k in range(len(acf))])
        peak = find_peaks(acf)[0]
        max_peak = peak[np.argsort(acf[peak])[-10:]][::-1]
        res.append(np.sort(tao[max_peak]))
        ax[1][i].scatter(tao[max_peak], acf[max_peak], marker='*', s=200, c='g',) #label=f"Period:{' '.join(f'{num:.2f}' for num in tao[max_peak])} Day")
        ax[1][i].scatter(tao, acf, s=3, c='r')

        frequency, power = LombScargle(x, y).autopower()
        period = 1 / frequency
        ax[2][i].plot(period, power, c='red')
        ax[2][i].set_xlim(0.5, 12)
        ax[2][i].set_ylim(0, 0.5)

        #ax[1][i].legend()
        print(f"{i}: {np.sort(tao[max_peak])}")

    plt.show()


    return res

def nrm(flux, intv):
    
    split_flux = np.split(np.array(flux), intv)

    target_flux = [(sf - np.mean(sf) + np.mean(flux)) for sf in split_flux]

    flux = np.concatenate(target_flux)

    return flux

for i in range(len(fits_file_target)):

    #if i not in np.where(dquality == 2)[0]: continue
    if i < 0: continue

    lc = Table.read(fits_file_target[i], format="fits")
    tic = lc['TIC'][0]

    tmag = float(tess[np.where(tess['TIC'] == tic)]['Tmag'])
    gaiadr3 = str(tess[np.where(tess['TIC'] == tic)]['DR3Name']).split('DR3 ')[1]

    #print(f"\n{i} TIC: {tic} Gaia DR3: {gaiadr3} \nPeriod:{period[i]} Data Quality:{dquality[i]}\n")
    
    #time, flux = get_flux(lc, tmag)
    #res = period_pathos(time, flux)

    #ipt = input("1: p | 2: dq | 3: fig")
    #if 'q' in ipt: break

    #if len(ipt) == 0: continue
    #dquality[i] = int(ipt[1])
    #if len(ipt) == 2: period[i] = res[2][int(ipt[0])-1]
    #elif len(ipt) == 3: period[i] = res[int(ipt[2])-1][int(ipt[0])-1]

    dr3.append(gaiadr3)

# %%
for tic in tess_result_vsini:
    lc = search_lightcurve(f"TIC {tic}", mission='TESS').download_all()

# %% 2 represent probable
for tic_id in tqdm(target_vsini[0][:10]):
    # 搜索光变曲线数据
    search_result = search_lightcurve(f'TIC {tic_id}',author='CDIPS', mission='TESS')

    # 下载数据
    lc = search_result.download_all().stitch()
    #print(lc)
    #continue

    # 剔除异常值
    lc_clean = lc.remove_outliers(sigma=2)
    print(tic_id)
    # 绘制光变曲线
    lc_clean.plot()
    plt.show()




# %%
for i,f in zip(np.where(dquality==2)[0], fits_file_target[np.where(dquality==2)[0]]):
    lc = Table.read(f, format="fits")
    tic = lc['TIC'][0]
    try:
        tmag = lc['Tmag'][0]
    except KeyError:
        tmag = tess[np.where(tess['TIC'] == lc['TIC'][0])]['Tmag']

    print(f"\n\n\n{i} {tic}\n\n\n")
    period_pathos(lc, tmag)

# %%
for f in fits_file_target[np.array([5,13])]:
    lc = Table.read(f, format="fits")
    tic = lc['TIC'][0]
    try:
        tmag = lc['Tmag'][0]
    except KeyError:
        tmag = tess[np.where(tess['TIC'] == lc['TIC'][0])]['Tmag']

    print(f"\n\n\n{i} {tic}\n\n\n")
    period_pathos(lc, tmag, cutoff=4000)

# %%
for i,f in enumerate(fits_file_target):
    lc = Table.read(f, format="fits")
    tic = lc['TIC'][0]

    index = np.where(cluster['TIC'] == tic)[0]

    cluster['Period'][index] = period[i]
    cluster['DQuality'][index] = dquality[i]




# %% Find out duplicate lightcurves
""" raw_tic_list = [int(x.split("tic-")[1].split("-s00")[0]) for x in folder_list]

find_duplicate_paths(raw_tic_list)
arr = pathos_list
item_counts = Counter(arr)

duplicates = [item for item, count in item_counts.items() if count > 1]
duplicate_indices = {}

for item in duplicates:
    indices = [index for index, value in enumerate(arr) if value == item]
    duplicate_indices[item] = indices

result = list(duplicate_indices.values())

print("重复元素为：", duplicates)
print("重复元素的位置为：", result)

# %% Exclude duplicate
for x in result:
    copy_files([folder_list[path] for path in x])

# %% FITS extend Tmag
for i, tic in enumerate(pathos_list[intersection]):
    num = np.where(tess["TIC"] == tic)[0]
    tmag = tess["Tmag"][num]
    
    process_fits_files(folder_list[intersection][i], tic, tmag)
    
# %%



# %%
for n in tqdm(range(len(folder_list[intersection]))):
    for f in find_fits_files(folder_list[intersection][n]):
        test = Table.read(f, format="fits")
        try:
            tic, mag = test['TIC'][0], test['Tmag'][0]
        except ZeroDivisionError():
            print(f)
        if pathos_list[intersection][n] != tic: print(f)
        if tess['Tmag'][np.where(tess['TIC'] == tic)[0]] != mag: print(f)


# %%
tess['vsini_ph']
 """
# %%


# %%

# %%
def period_cdips(lc, tmag, tic, cutoff=None):

    thres = 3
    figsize = (18, 8)
    

    cdips = search_lightcurve(f'TIC {tic}',author='CDIPS', mission='TESS').download_all().stitch()
    cdips = cdips.normalize().remove_outliers(sigma=thres)
    time, flux = np.array(cdips.time.value), np.array(cdips.flux.value)-1

    fig, ax = plt.subplots(2,3, figsize=figsize)
    cutoff = np.where(time > 801)[0][0]

    for i in range(3):

        if i == 0: x, y = time[:cutoff], flux[:cutoff] 
        elif i == 1: x, y = time[cutoff:], flux[cutoff:] 
        elif i==2: x, y = time, flux 

        ax[0][i].scatter(x, y, s=3, c='r')
        ax[0][i].set_title("CDIPS Normalized Lightcurve")

        acf = myfunc.acf(y, smooth=True, sigma=5)
        tao = np.array([(time[2] - time[1])*k for k in range(len(acf))])
        peak = find_peaks(acf)[0]
        max_peak = peak[np.argsort(acf[peak])[-5:]]
        ax[1][i].scatter(tao[max_peak], acf[max_peak], marker='*', s=200, c='g', label=f"Period:{tao[max_peak]} Day")
        ax[1][i].scatter(tao, acf, s=3, c='r')
        ax[1][i].set_xticks(np.arange(0, 6, 2))

    return 