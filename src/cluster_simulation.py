# %%
import myfunc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from functools import wraps
from loguru import logger
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

def validate_param(allowed_params):
    def decorator(func):
        @wraps(func)
        def wrapper(self, param):
            if param not in allowed_params:
                raise ValueError(f"Invalid parameter: {param}. Valid parameters are: {allowed_params}")
            return func(self, param)
        return wrapper
    return decorator

class Cluster:
    def __init__(self, cluster_id, cluster_data, lamost_data):
        self.cluster_id = cluster_id
        self.is_extinction = True
        self.is_emsto = False 
        self.cluster_data = cluster_data
        self.lamost_data = lamost_data
        self.cluster_color = self._set_cluster_color
        self.cluster_magnitude = self._set_cluster_magnitude
        self.lamost_color = self._set_lamost_color
        self.lamost_magnitude = self._set_lamost_magnitude

        self.mass = myfunc.replace_nans(self.lamost_data['Mass'])
        self.rad = myfunc.replace_nans(self.lamost_data['Rad'])
        self.age = self.lamost_data['age'][0]
        self.valid_rot_indice = self._valid_rot
        self.vsini = self.lamost_data['vsini']
        self.critical_vsini = (self.mass*myfunc.MSUN*myfunc.G/self.rad/myfunc.RSUN)**(1/2)/1000
        self.emsto_baseline_m = 30
        self.emsto_baseline_b = -10
        self.emsto_left_bound = -0.3
        self.emsto_right_bound = 0.3
        self.emsto_top_bound = -1.8
        self.emsto_bottom_bound = 1.8
        self.emsto_star = self._get_emsto_stars
        self.emsto_lamost_star = self._get_emsto_lamost_stars

    @property
    def _set_cluster_color(self):
        if self.is_extinction:
            return np.array(self.cluster_data['BP-RP_x']-self.cluster_data['Av_bp']+self.cluster_data['Av_rp'])
        else:
            return np.array(self.cluster_data['BP-RP_x'])
        
    @property
    def _set_cluster_magnitude(self):
        if self.is_extinction:
            return myfunc.magnitude_converter(np.array(self.cluster_data['BPmag']), 
                                              parallax_or_distance=self.cluster_data['rpgeo'])[0]    

        else:
            return myfunc.magnitude_converter(np.array(self.cluster_data['BPmag']-self.cluster_data['Av_bp']), 
                                              parallax_or_distance=self.cluster_data['rpgeo'])[0]

    @property
    def _set_lamost_color(self):
        if self.is_extinction:
            return np.array(self.lamost_data['BP-RP_x']-self.lamost_data['Av_bp']+self.lamost_data['Av_rp'])
        else:
            return np.array(self.lamost_data['BP-RP_x'])
        
    @property
    def _set_lamost_magnitude(self):
        if self.is_extinction:
            return myfunc.magnitude_converter(np.array(self.lamost_data['BPmag']), 
                                              parallax_or_distance=self.lamost_data['rpgeo'])[0]    

        else:
            return myfunc.magnitude_converter(np.array(self.lamost_data['BPmag']-self.lamost_data['Av_bp']), 
                                              parallax_or_distance=self.lamost_data['rpgeo'])[0]
        
    @property
    def _valid_rot(self):
        valid_mass_index, valid_rad_index = np.where(self.mass>0)[0], np.where(self.rad>0)[0]
        valid_crit_velocity_index = myfunc.intersection(valid_mass_index, valid_rad_index)[0]
        
        valid_vsini = np.where(self.lamost_data['vsini'] > -9999)[0]
        valid_rot_stars = myfunc.intersection(valid_vsini, valid_crit_velocity_index)[0]
        return valid_rot_stars

    @property
    def _get_emsto_stars(self):
        points_inside_bounds = np.array([
            i for i, (x, y) in enumerate(zip(self.cluster_color, self.cluster_magnitude)) \
            if self.emsto_left_bound <= x <= self.emsto_right_bound and \
                self.emsto_bottom_bound <= y <= self.emsto_top_bound
        ])
        return points_inside_bounds
    
    @property
    def _get_emsto_lamost_stars(self):
        points_inside_bounds = np.array([
            i for i, (x, y) in enumerate(zip(self.lamost_color, self.lamost_magnitude)) \
            if self.emsto_left_bound <= x <= self.emsto_right_bound and \
                self.emsto_bottom_bound <= y <= self.emsto_top_bound
        ])
        return points_inside_bounds

    def _setattr(self, name: str, value: np.any) -> None:
        if name in ['emsto_baseline_m', 'emsto_baseline_b']:
            logger.info(f"Setting {name} to {value}")
            super().__setattr__(name, value)
        else:
            logger.warning("attribute must be emsto_baseline_m or emsto_baseline_b")
            return 

    @validate_param(['mass', 'feh', 'vsini', 'crit_vsini', 'wi'])
    def _cluster_param_mapper(self, param):
        logger.info(f'Demonstrate the distribution of {param}')
        if param == 'mass':
            return {
                'value': self.mass[np.where(self.mass > 0)[0]],
                'bins': np.linspace(0.5, 3, 11),
                'text': 'Mass'
            }
        if param == 'feh':
            logger.info(np.mean(self.lamost_data['feh_1']), 
                        myfunc.feh_to_z(np.mean(self.lamost_data['feh_1'])))
            return {
                'value': self.mass[np.where(self.mass > 0)[0]],
                'bins': np.linspace(-1, 1, 11),
                'text': 'Mass'
            }
        if param == 'visni':
            return {
                'value': self.vsini[np.where(self.vsini>-9999)[0]],
                'bins': np.linspace(0, 500, 11),
                'text': 'Vsini'
            }
        if param == 'wi':
            return {
                'value': self.vsini[self.valid_rot_indice]/self.critical_vsini[self.valid_rot_indice],
                'bins': np.linspace(0, 1, 11),
                'text': 'Wi'
            }

    @classmethod
    def remove_ruwe(ruwe):
        kde = gaussian_kde(ruwe)
        xx = np.linspace(min(ruwe), max(ruwe), len(ruwe))
        density = kde(xx)

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
    
        return x[cut_off_index]    

    def plot_cmd(self):
        sns.set_style('whitegrid')
        plt.figure(figsize=(6,6))
        plt.scatter(self.cluster_color, self.cluster_magnitude, s=2, c='black')
        plt.scatter(self.lamost_color, self.lamost_magnitude, s=20, c='red', marker='*')

        print(f"Cluster Name: {self.cluster_id} \
          \nCluster Member: {len(self.cluster_data)} \
          \nCluster Member with Vsini: {len(self.critical_vsini)} \
          \nCluster Age: {self.age}" 
          )

        """ plt.text(np.max(lamost_x)*0.1, np.min(lamost_abs_mag_av[0])*0.1, f"Cluster Name: {c} \
            \nCluster Member: {len(cluster_data)} \
            \nCluster Member with Vsini: {len(valid_vsini)} \
            \nCluster Age: {lamost_data['age'][0]}" ) """
        
        plt.gca().invert_yaxis()
        plt.xlabel("BP_RP")
        plt.ylabel("BP")
        #plt.title(c)
        #plt.savefig(f"D:/Download/{c}_main_sequence.png")

        if not self.is_emsto: return

        plt.fill_between([self.emsto_left_bound, self.emsto_right_bound], self.emsto_top_bound, [self.emsto_bottom_bound,self.emsto_bottom_bound], \
                         color='red', alpha=0.3)
        plt.scatter(self.cluster_color[self.emsto_star], self.cluster_magnitude[self.emsto_star], \
                    s=20, c='green', marker='X')
        plt.plot(self.lamost_color[self.emsto_star], (self.emsto_baseline_m*self.cluster_color[self.emsto_star] \
                    - self.cluster_magnitude[self.emsto_star] + self.emsto_baseline_b), \
                    color='blue', label='baseline', lw=0.1)
        plt.ylim(-2, 10)

        plt.show()

        return 
        
    def plot_param_distrb(self, param):
        try:
            value, bins, text = self._cluster_param_mapper(param).values()
            myfunc.plot_kde(value, bins=bins, x_nrm=False, text=text)
        except OSError as e:
            logger.error(f'There is an issue during demonstrating the distribution of {param}...\n{e}')

    def cal_emsto_width(self):
        baseline_d = np.abs(self.emsto_baseline_m*self.cluster_color[self.emsto_star] - \
                            self.cluster_magnitude[self.emsto_star] + self.emsto_baseline_b) / \
                            np.sqrt(self.emsto_baseline_m**2 + 1)
        
        center_baseline_d = baseline_d - np.mean(baseline_d)

        myfunc.plot_kde(center_baseline_d, text='eMSTO Width')

        return

class ClusterISO(Cluster):
    def __init__(self, cluster, cluster_data, lamost_data):
        super().__init__(cluster, cluster_data, lamost_data)
        self.iso = 1

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

class ClusterMESA:
    def __init__(self, cluster_id, age, mesafile):
        self.cluster_id = cluster_id
        self.age = age
        self.mesafile = mesafile
        self.mesadata = np.genfromtxt(self.mesafile, names=True, skip_header=5)

        self.age_eep = self.mesadata['star_age']/10**8
        self.v_rot = self.mesadata['surf_avg_v_rot']
        self.v_rot_crit = self.mesadata['surf_avg_v_crit']
        self.v_eep = self.v_rot/self.v_rot_crit 
        self.omega_eep = self.mesadata['surf_avg_omega']/self.mesadata['surf_avg_omega_crit']
        self.mass_eep = self.mesadata['star_mass']
        #intv = [np.where(age_eep>0.5)[0][0],np.where(age_eep>8)[0][0]]
        self.nearest_age, self.nearest_age_index = self._find_nearest_age
        self.loglum = self.mesadata['log_L'][self.nearest_age_index]
        self.teff = self.mesadata['log_Teff'][self.nearest_age_index]

    @property    
    def _find_nearest_age(self):
        nearest_age_index = min(range(len(self.age_eep)), key=lambda i: abs(self.age_eep[i] - self.age))
        nearest_age = self.age_eep[nearest_age_index]
        return nearest_age, nearest_age_index

    def plot_fit_eep(self):
        sns.set_style('ticks') 

        _, initial_index = self.find_nearest_age(self.age_eep, 1)

        plt.scatter(self.age_eep, self.v_eep,
                    lw=2, 
                    #label=f'Wi EEP | Variation:{self.v_eep[self.nearest_age_index]-self.v_eep[self.initial_index]}', 
                    color='red')
        
        #plt.scatter(age, v_rot/300, lw=2, label='W EEP | Variation:{}'.\
        #            format(v_rot[nearest_age_index]-v_rot[initial_index]), color='green')
        #plt.scatter(age, v_rot_crit/600, lw=2, label='W_crit EEP | Variation:{}'.\
        #            format(v_rot_crit[nearest_age_index]-v_rot_crit[initial_index]), color='blue')
        plt.plot(self.age_eep[self.nearest_age_index], 
                 self.v_eep[self.nearest_age_index], 
                 marker='o', markersize=10, color='blue', label='Target Evolutionary Age', )
        #plt.plot(age, omega_fit, color='blue', label='Fitted Omega')     
        plt.text(self.age_eep[self.nearest_age_index]*0.9, 
                    self.v_eep[self.nearest_age_index]*0.9,
                    f'Ω = {self.v_eep[self.nearest_age_index]:.3f}', 
                    fontsize=15, ha='left', va='bottom', color='black')
        #plt.errorbar(age, omega, yerr=np.abs(omega_fit-omega), fmt='o', label="Original Data with Error")
        
        plt.xlabel('Age', fontweight='bold', size=14)
        plt.ylabel('Wi(W/W_crit)', fontweight='bold', size=14)
        #plt.title('EEP of Ω', size=15, fontweight='bold')
        legend = plt.legend()
        for text in legend.texts:
            text.set_weight('bold')
            text.set_fontsize(11)
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')

        plt.show()


