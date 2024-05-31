# %%
import myfunc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.optimize import curve_fit

# %%
def find_nearest_age(age_eep, age):
    nearest_age_index = min(range(len(age_eep)), key=lambda i: abs(age_eep[i] - age))
    nearest_age = age_eep[nearest_age_index]
    return nearest_age, nearest_age_index

def inv_cdf(u, alpha=2.35, xmin=3, xmax=4.9):
    return (u * (xmax**(1-alpha) - xmin**(1-alpha)) + xmin**(1-alpha))**(1/(1-alpha))

def gnr_star_mass_distrb():
    n_samples = 1000
    u_samples = np.random.rand(n_samples)
    mass_samples = np.round(inv_cdf(u_samples), decimals=1)
    return mass_samples

class Star:

    def __init__(self, mass, initial_omega) -> None:
        self.mass = mass
        self.initial_omega = initial_omega

        f = "D:/Spin/mesa/model/{}M/history_{}.data".format(self.mass, self.initial_omega) 
        data = np.genfromtxt(f, names=True, skip_header=5)
        self.age_eep = data['star_age']/10**8
        self.max_age = self.age_eep[-1]
        self.omega_eep = data['surf_avg_omega']/data['surf_avg_omega_crit']

        return

    def evolution(self, age):
        if age > self.max_age: 
            self.type = 'bad'
            return
        
        self.age = age
        nearest_age, nearest_age_index = find_nearest_age(self.age_eep, age)
        self.omega = self.omega_eep[nearest_age_index]

        if self.omega > 0.5: self.type = 'red'
        else: self.type = 'blue'

        return
    
def dsp_mass_distribution(mass, type, bins=100):
    plt.hist(mass, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Mass/Msun')
    plt.ylabel('Probability Density')
    plt.title('Mass Probability Density Function of {}'.format(type))
    plt.grid(True)
    plt.show()

def power_law(x, A, alpha):
    return A * x**(-alpha)

def fit_power_law(data, label, plot=True):
    hist, bins = np.histogram(data, bins=20, density=False)
    #print(hist)
    x = (bins[:-1] + bins[1:]) / 2
    valid = np.where(hist>0)[0]
    popt, pcov = curve_fit(power_law, x[valid], hist[valid])

    print(f"{label} Power Index = {popt[1]}")

    if not plot: return

    plt.hist(data, bins=bins, density=False, alpha=0.5, label=f'{label} Mass PDF')
    plt.plot(x, power_law(x, *popt), 'r-', label=f'{label} Mass Power Law Fitting')

    plt.xlabel('Mass')
    plt.ylabel('Star Number')
    plt.title(f'Power Law Fit to {label} Mass Distribution')
    plt.legend()

    plt.show()

    return

# %%
mass_distribution = gnr_star_mass_distrb()
cluster = []
for star_mass in tqdm(mass_distribution):
    star_omega = np.random.choice([0.4, 0.5, 0.6, 0.7])
    cluster.append(Star(star_mass, star_omega))

# %%
for evolution_age in range(17, 22):
    for star in cluster:
        star.evolution(evolution_age/10)

    bms, rms = [], []

    for star in cluster:
        if star.type == 'blue': bms.append(star.mass)
        elif star.type == 'red': rms.append(star.mass)

    print(f"Evolution age = {evolution_age*10} Myr")

    print(f"bms count = {len(bms)} | rms count = {len(rms)}")

    fit_power_law(bms, 'bms', plot=False)
    fit_power_law(rms, 'rms', plot=False)



# %%
