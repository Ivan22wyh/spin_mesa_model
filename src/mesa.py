import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.nearest_age, self.nearest_age_index = self.find_nearest_age(self.age_eep, self/age)
        self.loglum = self.mesadata['log_L'][self.nearest_age_index]
        self.teff = self.mesadata['log_Teff'][self.nearest_age_index]
        
    """ @property
    def _get_all_files(self):
        return sorted(glob(os.path.join(f"/home/ivan22/MESA/models/{self.cluster_id.remove('_').lower()}", \
                                        f'**/*.data'), recursive=True)) """

    def find_nearest_age(self):
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


