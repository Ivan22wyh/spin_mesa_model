# %%
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
iso = np.zeros((10, 100))

age = 8.2
coff_omega = {}
line_head = "#      LOGL      LOG_TE      MASS      Z      MDOT      OMEGA"
bc_table = [line_head]

# %%
def list_files(path, ftype=None):
    files = []
    for r, d, f in os.walk(path):
        files.extend(os.path.join(r, file) for file in f)

    files = np.array([f.replace('\\', '/') for f in files])

    if ftype: files = [f for f in files if f".{ftype}" in f]

    return np.array(files)

def find_nearest_age(age_eep, age):
    nearest_age_index = min(range(len(age_eep)), key=lambda i: abs(age_eep[i] - age))
    nearest_age = age_eep[nearest_age_index]
    return nearest_age, nearest_age_index

def list_to_dat(data_list, filename):
    # 将列表写入.dat文件
    with open(filename, 'w') as f:
        for item in data_list:
            f.write(str(item) + '\n')

def write_bc(data, omega, nearest_age_index, age = 4.5*10**8, z=0.012206883460363242, inclination='normal'):
    if inclination == 'normal':
        target_logL = data['log_L'][nearest_age_index]
        target_logteff = data['log_Teff'][nearest_age_index]
    if inclination == 'polar':
        target_logL = np.log10(data['grav_dark_L_polar'])[nearest_age_index]
        target_logteff = np.log10(data['grav_dark_Teff_polar'])[nearest_age_index]
    if inclination == 'equatorial':
        target_logL = np.log10(data['grav_dark_L_equatorial'])[nearest_age_index]
        target_logteff = np.log10(data['grav_dark_Teff_equatorial'])[nearest_age_index] 
    target_mass = data['star_mass'][nearest_age_index]
    target_mdot = data['star_mdot'][nearest_age_index]
    line = f'{target_logL}      {target_logteff}      {target_mass}      {z}      {target_mdot}      {omega}      {inclination}      '
    
    bc_table.append(line)

def plot_fit_eep(age, omega, v, v_rot, v_rot_crit, mass, nearest_age_index):
    sns.set_style('ticks') 

    #plt.scatter(age, omega, lw=2, label='EEP', color='red')
    _, initial_index = find_nearest_age(age, 1)

    plt.scatter(age, v, lw=2, label=f'Wi EEP | Variation:{v[nearest_age_index]-v[initial_index]}', color='red')
    #plt.scatter(age, v_rot/300, lw=2, label='W EEP | Variation:{}'.\
    #            format(v_rot[nearest_age_index]-v_rot[initial_index]), color='green')
    #plt.scatter(age, v_rot_crit/600, lw=2, label='W_crit EEP | Variation:{}'.\
    #            format(v_rot_crit[nearest_age_index]-v_rot_crit[initial_index]), color='blue')
    plt.plot(age[nearest_age_index], v[nearest_age_index], marker='o', markersize=10, color='blue', label='Target Evolutionary Age', )
    #plt.plot(age, omega_fit, color='blue', label='Fitted Omega')     
    plt.text(age[nearest_age_index]*0.9, v[nearest_age_index]*0.9, \
                 f'Ω = {omega[nearest_age_index]:.3f}', fontsize=15, ha='left', va='bottom', color='black')
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

def main(cluster, mass, plot_range):
    for f in list_files(f'D:/Spin/mesa/model/{cluster}/{mass}M')[plot_range[0]:plot_range[1]]:
        omega = f.split('.data')[0].split('history_')[1]
        print(f"Analyzing EEP of Omega = {omega}")
        
        data = np.genfromtxt(f, names=True, skip_header=5)
        age_eep = data['star_age']/10**8
        v_eep = data['surf_avg_v_rot']/data['surf_avg_v_crit']
        omega_eep = data['surf_avg_omega']/data['surf_avg_omega_crit']
        mass_eep = data['star_mass']
        #intv = [np.where(age_eep>0.5)[0][0],np.where(age_eep>8)[0][0]]
        nearest_age, nearest_age_index = find_nearest_age(age_eep, age)
        print(len(v_eep))
        print(data['log_L'][nearest_age_index], data['log_Teff'][nearest_age_index])

        plot_fit_eep(age=age_eep,
                    omega=data['surf_avg_omega_div_omega_crit'],
                    v_rot=data['surf_avg_v_rot'],
                    v_rot_crit=data['surf_avg_v_crit'],
                    v=v_eep,
                    mass=mass_eep,
                    nearest_age_index=nearest_age_index,
                    )    


if __name__ == '__main__':
    #main('ngc1647', 1.2, (0, 8))   
    main('ngc6811', 1.1, (8, 9))

# %%

