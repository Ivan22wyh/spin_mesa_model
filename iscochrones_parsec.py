# %%
import myfunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import gaussian_kde
from torch import Value
from tqdm import tqdm
from scipy.interpolate import interp1d
import warnings
import seaborn as sns

# 禁用所有警告
warnings.filterwarnings("ignore")


def remove_ruwe(ruwe):

    # 使用高斯核密度估计
    kde = gaussian_kde(ruwe)

    # 生成一组值，用于评估估计的密度
    xx = np.linspace(min(ruwe), max(ruwe), len(ruwe))

    # 计算估计的密度值
    density = kde(xx)

    # 绘制原始数据的直方图
    #plt.hist(ruwe, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')
    """ 
    # 绘制高斯核密度估计的曲线
    plt.plot(xx, density, linewidth=2, color='red', label='Real Cluster RUWE Distribution')

    plt.xlim(0.75, 2)
    plt.xlabel('RUWE')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Gaussian Kernel Density Estimation of Gaia DR3 RUWE') """
    

    peak_index = np.argmax(density)
    xrange = xx[peak_index] - xx[0]
    reflected_x = xx[:peak_index] + xrange
    reflected_pdf = density[:peak_index][::-1]

    x = np.hstack((xx[:peak_index], reflected_x))
    pdf = np.hstack((density[:peak_index], reflected_pdf))
    normalized_pdf = pdf / np.trapz(pdf, x)

    cdf_values = np.cumsum(normalized_pdf)/np.sum(normalized_pdf)
    threshold = 0.99
    cut_off_index = np.argmax(cdf_values >= threshold)
    """     
    plt.plot(x, pdf, linewidth=3, label='Single Star RUWE Distribution')

    plt.axvline(x=x[cut_off_index], color='black', linestyle='--', label="99% Cutoff")
    plt.axvline(1, color='grey', linestyle='--', label="Reflected axis")

    plt.legend()
    plt.show() """
    
    return x[cut_off_index]

def cal_emsto_width(m, b, x, y):
    baseline_d = np.abs(m*x - y + b) / np.sqrt(m**2 + 1)
    center_baseline_d = baseline_d - np.mean(baseline_d)

    #myfunc.plot_kde(center_baseline_d)
    myfunc.plot_kde(baseline_d)

    return

def get_eMSTO_stars(x, y, left_bound, right_bound, top_bound, bottom_bound):
    points_inside_bounds = []
    for x, y in zip(x, y):
        if left_bound <= x <= right_bound and bottom_bound <= y <= top_bound:
            points_inside_bounds.append((x, y))
    points_inside_bounds = np.array(points_inside_bounds)
    return points_inside_bounds[:, 0], points_inside_bounds[:, 1]

def crit_rot(lamost_data):
    mass = myfunc.replace_nans(lamost_data['Mass'])
    rad = myfunc.replace_nans(lamost_data['Rad'])
    valid_mass_index, valid_rad_index = np.where(mass>0)[0], np.where(rad>0)[0]
    crit_velocity = (mass*myfunc.MSUN*myfunc.G/rad/myfunc.RSUN)**(1/2)/1000
    valid_crit_velocity_index = myfunc.intersection(valid_mass_index, valid_rad_index)[0]
    
    return crit_velocity, valid_crit_velocity_index

def connect_iso(x, y):
    # 将点按照 x 坐标进行排序
    sorted_indices = sorted(range(len(x)), key=lambda i: x[i])
    sorted_x = [x[i] for i in sorted_indices]
    sorted_y = [y[i] for i in sorted_indices]

    # 绘制散点图
    plt.scatter(sorted_x, sorted_y)

    # 连接点
    for i in range(len(sorted_x) - 1):
        plt.plot([sorted_x[i], sorted_x[i+1]], [sorted_y[i], sorted_y[i+1]], color='blue')

    # 显示连接的图形
    plt.show()
    
def extract_iso(filename, omega):
    bp, rp = [], []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):
            if '{} '.format(omega) in lines[i]:
                #print(lines[i].strip())
                bp.append(lines[i+1].strip().split( )[2])
                rp.append(lines[i+1].strip().split( )[3])
    
    bp, rp = np.array(bp, dtype=float), np.array(rp, dtype=float)
    return bp, bp-rp

def set_age(fits_file, cluster, age):
    fits_file['age'][np.where(fits_file['Cluster']==cluster)[0]] = age

    return 

def interpolate_iso(x_points, y_points, omega):
    # 创建拉格朗日插值函数
    f = interp1d(x_points, y_points, kind='cubic')

    # 生成更密集的 x 值用于绘图
    x_new = np.linspace(min(x_points), max(x_points), 100)
    
    # 计算对应的 y 值
    y_new = f(x_new)

    # 绘制原始点
    plt.scatter(x_points, y_points)
    
    # 绘制插值曲线
    plt.plot(x_new, y_new, )

    '''for i in range(len(x_points)):
        # 以插值点为基准，获取斜率
        slope = (f(x_points[i] + 0.001) - f(x_points[i] - 0.001)) / 0.002
        if slope != 0:
            # 计算垂直方向上的随机点
            delta_x = np.random.uniform(-0.05, 0.05, 3)
            delta_y = delta_x / slope
            random_x = x_points[i] + delta_x
            random_y = y_points[i] + delta_y
            plt.plot(random_x, random_y, 'go')'''

    return


def extract_iso(filename, omega):
    bp, rp = [], []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):
            if '{} '.format(omega) in lines[i]:
                #print(lines[i].strip())
                bp.append(lines[i+1].strip().split( )[2])
                rp.append(lines[i+1].strip().split( )[3])
    
    bp, rp = np.array(bp, dtype=float), np.array(rp, dtype=float)
    return bp, bp-rp

# 调用函数并传入.dat文件的路径
#sns.set_style('whitegrid')
#plt.figure(figsize=(6,6))

thres = 0.5
cluster = Table.read("resources/open_cluster_member_v8.fits", format="fits")
cluster_lamost_afgk = Table.read("resources/wyh_afgk_v4.fits", format="fits")
cluster_lamost = Table.read("resources/wyh_v5.fits", format="fits")
cluster_lamost_mrs = Table.read("resources/wyh_mrs_v4.fits", format="fits")
cluster = cluster[np.where((cluster['Proba'] > thres))]
cluster_lamost = cluster_lamost[np.where((cluster_lamost['Proba'] > thres))]
cluster = cluster[np.where(cluster['isbinary']==0)[0]]
cluster_lamost = cluster_lamost[np.where(cluster_lamost['isbinary']==0)[0]]

cluster['RUWE'][np.isnan(np.array(cluster['RUWE']))] = 1

count = myfunc.count_element(cluster_lamost['Cluster'])
#count = myfunc.count_element(cluster['Cluster'])
valid_cluster = [(c,n) for c,n in count if n > 30]

# %% Draw DR3 cluster cmd
extended = np.load("extended.npz")
final_cluster = np.load("final_final_cluster.npz")['cluster']

for c, n in valid_cluster[:]:
    # Choose the valid cluster
    if c in final_cluster: continue
    #if c not in extended['non']: continue
    if c != "NGC_1528": continue

    cluster_data = cluster[np.where(cluster['Cluster'] == c)[0]]
    cluster_x = np.array(cluster_data['BP-RP_x']-cluster_data['Av_bp']+cluster_data['Av_rp'])
    abs_mag = myfunc.magnitude_converter(np.array(cluster_data['BPmag']), parallax_or_distance=cluster_data['rpgeo'])    
    abs_mag_av = myfunc.magnitude_converter(np.array(cluster_data['BPmag']-cluster_data['Av_bp']), parallax_or_distance=cluster_data['rpgeo'])

    lamost_data = cluster_lamost[np.where(cluster_lamost['Cluster'] == c)[0]]
    lamost_x = np.array(lamost_data['BP-RP_x']-lamost_data['Av_bp']+lamost_data['Av_rp'] )
    lamost_abs_mag = myfunc.magnitude_converter(np.array(lamost_data['BPmag']), parallax_or_distance=lamost_data['rpgeo'])
    lamost_abs_mag_av = myfunc.magnitude_converter(np.array(lamost_data['BPmag']-lamost_data['Av_bp']), parallax_or_distance=lamost_data['rpgeo'])
    
    valid_vsini = np.where(lamost_data['vsini'] > 52)[0]
    crit_velocity, valid_crit_velocity_index = crit_rot(lamost_data)
    valid_rot_stars = myfunc.intersection(valid_vsini, valid_crit_velocity_index)[0]
    if len(valid_vsini) < 30: continue

    # Plot CMD of the cluster
    print(f"Cluster Name: {c} \
          \nCluster Member: {len(cluster_data)} \
          \nCluster Member with Vsini: {len(valid_vsini)} \
          \nCluster Age: {lamost_data['age'][0]}" 
          )


    #continue
    sns.set_style('whitegrid')
    plt.figure(figsize=(6,6))
    plt.scatter(cluster_x, abs_mag_av[0], s=2, c='black')
    plt.scatter(lamost_x, lamost_abs_mag_av[0], s=20, c='red', marker='*')
    plt.text(np.max(lamost_x)*0.1, np.min(lamost_abs_mag_av[0])*0.1, f"Cluster Name: {c} \
          \nCluster Member: {len(cluster_data)} \
          \nCluster Member with Vsini: {len(valid_vsini)} \
          \nCluster Age: {lamost_data['age'][0]}" )
    for i in [3, 8]:
        bp, bp_rp = extract_iso('D:/Spin/resources/output_YBC.txt', i/10)
        #plt.gca().invert_yaxis()
        #interpolate_iso(bp_rp, bp, i/10, label)
        #plt.plot(bp_rp, bp, label='W={}'.format(i/10))


    # Calculate eMSTO width
    #m, b = 30, -10
    #left_bound, right_bound, top_bound, bottom_bound = -0.3, 0.3, 1.8, -1.8
    #emsto_x, emsto_y = get_eMSTO_stars(cluster_x, abs_mag_av[0], left_bound, right_bound, top_bound, bottom_bound)
    #emsto_vsini_x, emsto_visni_y = get_eMSTO_stars(lamost_x, lamost_abs_mag_av[0], left_bound, right_bound, top_bound, bottom_bound)
    
    #plt.fill_between([left_bound, right_bound], top_bound, [bottom_bound,bottom_bound], color='red', alpha=0.3)
    #plt.scatter(emsto_x, emsto_y, s=20, c='green', marker='X')
    #plt.plot(lamost_x, m*lamost_x+b, color='blue', label='baseline', lw=0.1)
    #plt.ylim(-2, 10)

    #plt.xlim(-0.3, 0.8)
    #plt.ylim(-1, 5)
    plt.gca().invert_yaxis()
    plt.xlabel("BP_RP")
    plt.ylabel("BP")
    plt.title(c)
    plt.savefig(f"D:/Download/{c}_main_sequence.png")

    plt.show()

    #cal_emsto_width(m, b, emsto_x, emsto_y)


    # Calculate centered FEH
    try:
        print(np.mean([x for x in lamost_data['feh_1']]), myfunc.feh_to_z(np.mean([x for x in lamost_data['feh_1']]),))
        break
    except:
        print('no feh')
    #myfunc.plot_kde([x for x in lamost_data['feh_1']], bins=np.linspace(np.min(lamost_data['feh_1']), np.max(lamost_data['feh_1']), 21),)
    myfunc.plot_kde([x for x in lamost_data['Mass'] if x>0], bins=np.linspace(0.5, 3, 11), text='Mass')
    #continue
    try:
        myfunc.plot_kde(lamost_data[valid_rot_stars]['vsini']/crit_velocity[valid_rot_stars],
                    bins=np.linspace(0, 1, 11),
                    x_nrm=False,
                    save=f"D:/Download/{c}_vsini.png",
                    text='rotation'
                )
    except:
        pass



    #if input('') == 'q': break

    #break

    

#print("Finally Valid Cluster(enough vsini count): {}".format(len(analysis_cluster)))










# %% add vsini
cluster_lamost = Table.read("resources/wyh.fits", format="fits")
cluster_lamost['vsini'] = 0.0
df = pd.read_csv("resources/wyh_0418.csv")
vsini = df['vsini_ph']
for i in range(len(vsini)):
    if vsini[i] == 0.0:
        if np.random.rand() > 0.5: res = 0.0
        else: res = 51
    else: res = vsini[i] 
    cluster_lamost['vsini'][i] = res
cluster_lamost.write("resources/wyh_v2.fits", format="fits", overwrite=True)

# %%
cluster_lamost = Table.read("resources/wyh_v2.fits", format="fits")
len(np.where(cluster_lamost['vsini'] == 51.0)[0])

# %%
