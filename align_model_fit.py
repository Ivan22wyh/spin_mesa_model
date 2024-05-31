# %%
import emcee
import myfunc
import numpy as np
from scipy import stats
from scipy.stats import norm, t, kde
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.table import Table
import corner

def sini_func(Rn0, Rn1, alpha, lambda_): 
    phi = 2*np.pi*Rn1
    theta = np.arccos(1-Rn0*(1-np.cos(lambda_)))
    cosi = np.sin(alpha) * np.sin(theta) * np.cos(phi) + np.cos(alpha) * np.cos(theta)
    sini = np.sin(np.arccos(np.abs(cosi)))
    return sini

def double_peak_sin_func(rn0, rn1, f, alpha1, lambda_1, alpha2, lambda_2):
    phi = 2*np.pi*rn1
    if np.random.uniform(0, 1) >= f: 
        alpha, lambda_ = alpha1, lambda_1
    else:
        alpha, lambda_ = np.random.uniform(0, 1.57), 1.57
    theta = np.arccos(1-rn0*(1-np.cos(lambda_)))
    cosi = np.sin(alpha) * np.sin(theta) * np.cos(phi) + np.cos(alpha) * np.cos(theta)
    sini = np.sin(np.arccos(np.abs(cosi)))
    return sini

def sini_func_f(rn0, rn1, f, alpha1, lambda_1):
    phi = 2*np.pi*rn1
    if np.random.uniform(0, 1) >= f: 
        alpha, lambda_ = alpha1, lambda_1
    else:
        alpha, lambda_ = alpha1, 90/180*np.pi
    theta = np.arccos(1-rn0*(1-np.cos(lambda_)))
    cosi = np.sin(alpha) * np.sin(theta) * np.cos(phi) + np.cos(alpha) * np.cos(theta)
    sini = np.sin(np.arccos(np.abs(cosi)))
    return sini

def sim_sini(alpha, lambda_, rn0=np.random.rand(), rn1=np.random.rand(), num=1000):
    sini = np.array([sini_func(rn0, rn1, alpha, lambda_) for x in range(num)])
    sini = sini[np.where(sini >= 0.2)[0]]
    length = len(sini)

    std1, std2 = 0.1, 0.1
    u1, u2 = np.random.normal(loc=0, scale=1, size=length), np.random.normal(loc=0, scale=1, size=length)
    res = sini*((1+std1*u1)/(1+std2*u2))
 
    return res

def double_peak_sim_sini(alpha1, lambda_1, alpha2, lambda_2,  f, rn0=np.random.rand(), rn1=np.random.rand(), num=1000):
    sini = np.array([double_peak_sin_func(rn0, rn1, f, alpha1, lambda_1, alpha2, lambda_2) for x in range(1000)])
    length = len(sini)

    std1, std2 = 0.1, 0.1
    u1, u2 = np.random.normal(loc=0, scale=1, size=length), np.random.normal(loc=0, scale=1, size=length)
    res = sini*((1+std1*u1)/(1+std2*u2))

    return res

def plot_sini_cdf(sini, label):

    kde_func = kde.gaussian_kde(sini)
    x_values = np.linspace(0, 1.5, 1000)
    pdf_values = kde_func(x_values)
    cdf_values = np.cumsum(pdf_values) / np.sum(pdf_values)

    """
    # 绘制平滑的 PDF
    plt.subplot(121)
    plt.plot(x_values, pdf_values,c='red', lw=2, label=f'α=30°\nλ=10°')
    plt.title('Sini Probability Density Function')
    plt.xlabel('Sini')
    plt.ylabel('Probability')
    plt.legend(loc="upper left")   
    

    # 绘制平滑的 CDF
    """
    plt.subplot(122)
    plt.plot(x_values, cdf_values, lw=2,c='red', label=label)
    plt.title('Sini Cumulative Distribution Function')
    plt.xlabel('Sini')
    plt.ylabel('Probability')
    plt.legend(loc="lower right") 
    
    plt.tight_layout()
     

def plot_mcmc_heatmap(samples, parameter_ranges, parameter_names=None, bins=100, show=True):
    """
    绘制MCMC采样结果的参数概率密度热力图。

    参数：
    samples: 包含MCMC采样结果的数组, 每列对应一个参数。
    parameter_ranges: 参数的取值范围，应为一个列表，每个元素包含参数的取值范围 [min_value, max_value]。
    parameter_names: 参数名称的列表，用于标记图表上的轴。如果未提供，将使用默认标签。
    bins: 离散化取值范围时的网格数量。
    show: 是否显示图表。

    返回: None
    """
    num_parameters = samples.shape[1]

    if parameter_names is None:
        parameter_names = [f"Parameter {i+1}" for i in range(num_parameters)]

    # 创建一个热力图矩阵
    heatmap_matrix = np.zeros((bins, bins))

    for i in range(num_parameters):
        param_samples = samples[:, i]
        min_value, max_value = parameter_ranges[i]

        # 离散化参数取值
        param_bins = np.linspace(min_value, max_value, bins+1)
        param_centers = 0.5 * (param_bins[:-1] + param_bins[1:])
        counts, _ = np.histogram(param_samples, bins=param_bins)
        pdf = counts / (np.sum(counts) * (param_centers[1] - param_centers[0]))

        # 将概率密度填入热力图矩阵
        heatmap_matrix[i, :] = pdf

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_matrix, extent=(parameter_ranges[0][0], parameter_ranges[0][1], parameter_ranges[1][0], parameter_ranges[1][1]), origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Probability Density")
    plt.xlabel(parameter_names[0])
    plt.ylabel(parameter_names[1])
    plt.title("MCMC Parameter Probability Density Heatmap")

    if show:
        plt.show()


# %% Measured sini distribution
raw_i = np.random.normal(np.pi/3, 0.5, 1000)
i = raw_i[np.where((raw_i >= 0) & (raw_i <= np.pi/2))[0]]
length = len(i)
sini = np.sin(i)


random_v = np.random.uniform(100, 1000, size=length)
random_ratio = np.random.uniform(0.9, 1.1, size=length)

random_vsini = random_v * random_ratio * sini



# %% theoretical sini cdf
plt.figure(figsize=(8, 4))
rn0, rn1 = np.random.rand(), np.random.rand()
alpha1, alpha2 = 30/180*np.pi, 20/180*np.pi
lambda_1, lambda_2 = 60/180*np.pi, 20/180*np.pi
alphas, lambdas = [n*20/180*np.pi for n in [1, 2, 3]], [n*10/180*np.pi for n in [1, 5, 9]]



"""
for alpha in alphas:
    sini_distribution = sim_sini(alpha, lambda_1)
    plot_sini_cdf(sini_distribution, label=alpha) 

""" 
for f in [0.5,]:
    sini = double_peak_sim_sini(alpha1, lambda_1, alpha2, lambda_2, f=0.5, num=10000)
    plot_sini_cdf(sini, label=f"α1=30° λ1=20°\nα2=60° λ2=20°\n        f=0.5") 

plt.show()

# %%


















# %% mcmc拟合倾角分布参数

# 生成符合你的二元函数的随机数据
np.random.seed(0)  # 设置随机数种子以保持可重复性
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
data = double_peak_sin_func(X, Y, f=0.8, 
                            alpha1=45/180*np.pi, 
                            lambda_1=20/180*np.pi, 
                            ) #+ np.random.normal(0, 0.1, X.shape)

# 将二元数据展平
data_flat = data.flatten()

std1, std2 = 0.6, 0.6
u1, u2 = np.random.normal(loc=0, scale=1, size=len(data_flat)), np.random.normal(loc=0, scale=1, size=len(data_flat))
data_flat = data_flat*((1+std1*u1)/(1+std2*u2))

# 定义似然函数
def ln_likelihood(params):
    param1, param2= params
    expected_data = sini_func(X, Y, param1, param2)
    ln_like = -0.5 * np.sum((data_flat - expected_data.flatten())**2)
    return ln_like

# 定义先验函数
def ln_prior(params):
    param1, param2 = params
    # 在合理的参数范围内给予先验概率1，否则为0
    if 0 < param1 < 1.57 and 0 < param2 < 1.57:
        return 0.0
    return -np.inf

# 定义后验概率函数
def ln_posterior(params):
    ln_prior_value = ln_prior(params)
    if not np.isfinite(ln_prior_value):
        return -np.inf
    return ln_prior_value + ln_likelihood(params)

# 设置参数的初始值
initial_params = [30/180*np.pi, 30/180*np.pi]
ndim = len(initial_params)

# 设置MCMC采样的步数和步长
nwalkers = 20  # 行走者数量
nsteps = 5000  # 步数
burn_in = 1000  # 燃烧期

# 初始化行走者的位置
pos = [initial_params + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]

# 运行MCMC采样
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior)
sampler.run_mcmc(pos, nsteps, progress=True)

# 提取采样后的参数
samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))

# 绘制参数的corner图
labels = ['α', 'λ']  # 参数的标签

#samples = samples/np.pi*180

fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12}, cmap="viridis")

""" plt.show()

fig, ax = plt.subplots()
h = ax.hist2d(samples[:, 0], samples[:, 1], bins=50, cmap="bwr")
plt.colorbar(h[3], ax=ax)
plt.xlim(0, 1.57)
plt.ylim(0, 1.57) 
ax.set_xlabel("α")
ax.set_ylabel("λ") """



plt.show()
 # %% 速度分布计算
import numpy as np

import matplotlib.pyplot as plt

# 测量数据 A 和 B
data_A = np.random.uniform(1, 2, size=1000)
data_B = np.random.normal(5, 1, size=1000)
data_B = data_B*data_A

# 定义似然函数
def log_likelihood(theta, A, B):
    C = theta
    residuals = C - (A / B)
    return -0.5 * np.sum(residuals**2)

# 定义先验函数
def log_prior(theta):
    C = theta
    return -0.5 * (C**2)

# 定义后验函数
def log_posterior(theta, A, B):
    log_prior_value = log_prior(theta)
    log_likelihood_value = log_likelihood(theta, A, B)
    return log_prior_value + log_likelihood_value

# 执行 MCMC 采样
ndim = 1
nwalkers = 100
initial_pos = np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(data_A, data_B))
nsteps = 1000
sampler.run_mcmc(initial_pos, nsteps)

# 获取参数的后验分布
burn_in = 200
samples = sampler.chain[:, burn_in:, :].reshape(-1, ndim)

# 绘制后验分布
fig = corner.corner(samples, labels=['C'])
plt.show()


























# %% 极坐标实例
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据

sini1 = np.array([sini_func(np.random.rand(), np.random.rand(), 60/180*np.pi, 2/180*np.pi) for x in range(500)])
sini2 = np.array([sini_func(np.random.rand(), np.random.rand(), 60/180*np.pi, 10/180*np.pi) for x in range(500)])
sini3 = sim_sini(60/180*np.pi, 10/180*np.pi, num=500)
sini4 = np.array([sini_func(np.random.rand(), np.random.rand(), 60/180*np.pi, 90/180*np.pi) for x in range(500)])

sini_list = [sini1, sini2, sini3, sini4]

#theta = np.arcsin(np.concatenate((sini_list[2], sini_list[3])))
theta = np.arcsin(sini_list[1])
r = np.random.uniform(0.1, 1, size=len(theta))  # 极径

# 创建极坐标子图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# 绘制散点图
ax.scatter(theta, r, c='r', label='Spin Axis', s=10)

# 绘制带阴影的扇形
start_angle = np.pi*50/180
end_angle = np.pi*70/180
ax.fill_between([start_angle, end_angle], 0, 1, color='gray', alpha=0.3)

""" start_angle = np.pi*50/180
end_angle = np.pi*70/180
ax.fill_between([start_angle, end_angle], 0, 1, color='gray', alpha=0.3) """

# 设置图形属性
ax.set_title('Theoretical Spin Orientation Distribution(Alignment)')
#ax.set_xticklabels([])  # 隐藏极坐标刻度标签

# 设置坐标范围
ax.set_xlim(0, 0.5*np.pi)
ax.set_ylim(0, 1)

# 显示图例
ax.legend()

# 展示图形
plt.show()


    
# %%
thres = 0.5
cluster = Table.read("resources/open_cluster_member_v8.fits", format="fits")
cluster_lamost = Table.read("resources/cluster_pro_v8.fits", format="fits")
cluster = cluster[np.where((cluster['Proba'] > thres))]
cluster_lamost = cluster_lamost[np.where((cluster_lamost['Proba'] > thres))]
cluster = cluster[np.where(cluster['isbinary']==0)[0]]
cluster_lamost = cluster_lamost[np.where(cluster_lamost['isbinary']==0)[0]]
cluster_rot = np.load('ngc2548.npz')

# %%
cluster_rot = np.load('ngc2548.npz')
p = cluster_rot['p']
p_dq = cluster_rot['dq']
dr3 = cluster_rot['dr3']

for i in range(len(cluster_lamost)):
    obj = str(cluster_lamost[i]['DR3Name']).split('DR3 ')[1]
    if (cluster_lamost[i]['Cluster'] == 'NGC_2548') and \
        (obj in dr3): 
        index = np.where(dr3 == obj)[0]
        cluster_lamost[i]['Period'] = p[index]
        cluster_lamost[i]['PQuality'] = p_dq[index]

# %%
lamost_data = np.array(cluster_lamost[np.where(cluster_lamost['Cluster'] == 'NGC_2548')[0]])

r = np.array(lamost_data['Rad'])
r_err = myfunc.replace_nans(np.array(lamost_data['s_Rad']), value=0)
vsini = np.array(lamost_data['vsini_ph'])
vsini_err = np.array(lamost_data['vsini_ph_err'])
p = np.array(lamost_data['Period'])
p_dq = np.array(lamost_data['PQuality'])

#ints = myfunc.intersection(np.array([str(x).split('DR3 ')[1].split("'")[0] for x in lamost_data['DR3Name']]), dr3)


# %%
sini = vsini*1000*p*24*60*60/2/np.pi/r/myfunc.RSUN
sini_err = ((p/2/np.pi/r*vsini_err)**2 + (vsini*p/2/np.pi/r**2*r_err)**2)**(1/2)

sini

# %%
