import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# parameters
H = 2  # Hill coefficient
betay = 1
betaz = 1
alphay = 1  # a_deg + a_dil
alphaz = 1  
By = 0  # Basal expression of y, z
Bz = 0
Kxy = 0.1  # equilibrium constants
Kxz = 0.1
Kyz = 0.5

def f_activator(u, K, H):
    return (u/K)**H / (1 + (u/K)**H)
def f_repressor(u, K, H):
    return 1 / (1 + (u/K)**H) 
# competive 
def fc_activator(u, Ku, Kv, v, H):
    return (u / Ku)**H / (1 + (u/Ku)**H + (v/Kv)**H)
def fc_repressor(u, Ku, Kv, v, H):
    return 1 / (1 + (u/Ku)**H + (v/Kv)**H)    

def ODE_Y(t, initial_values, By, betay, Kxy, H, alphay):
    x, y, z = initial_values  # entpacken von x,y,z
    x = Sx(t)
    dydt = By + betay * f_activator(x, Kxy, H) - alphay * y
    return [dydt]

def ODE_Z(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
    x, y, z = initial_values
    x = Sx(t)
    y = np.interp(t, solution_y.t, solution_y.y[-1]) # um y kontinuierlich zu machen
    dzdt = Bz + betaz * f_activator(x, Kxz, H) * f_activator(y, Kyz, H) - alphaz * z
    return [dzdt]

def simp_reg(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
    x, y, z = initial_values
    x = Sx(t)
    y = 1
    dzdt = Bz + betaz * f_activator(x, Kxz, H) * f_activator(y, Kyz, H) - alphaz * z
    return [dzdt]

# Zeitschalter
def Sx(t):
    if 1 < t < 6:
        return 1
    else:
        return 0


# Zeitbereich 
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

initial_values = [1, By, Bz]

solution_y = solve_ivp(ODE_Y, t_span, initial_values, t_eval=t_eval, method = 'Radau', args=(By, betay, Kxy, H, alphay)) # RK45, RK23, DOP853, Radau
print(np.max(solution_y.y[-1]))
y_data = solution_y.y[-1]
solution_z = solve_ivp(ODE_Z, t_span, initial_values, t_eval=solution_y.t, args=(Bz, betaz, Kxz, Kyz, H, alphaz))
solution_simp_reg = solve_ivp(simp_reg, t_span, initial_values, t_eval=solution_y.t, args=(Bz, betaz, Kxz, Kyz, H, alphaz))


# Plotten
fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [0.3, 0.7]})

Sx = [Sx(t) for t in t_eval]
axs[0].plot(t_eval, Sx, label='Sx', color='purple')
axs[0].set_ylabel('Sx')
axs[0].set_xticks([])
axs[0].set_yticks(np.arange(0, 1.1, 1)) #start,stop,step
axs[0].legend()

axs[1].axvline(x=1, color='k', linestyle='--', linewidth=1)
axs[1].axvline(x=6, color='k', linestyle='--', linewidth=1)
axs[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
axs[1].axhline(y=1, color='k', linestyle='--', linewidth=1)
# axs[1].plot(solution_y.t, solution_y.y[-1], label='Y')
axs[1].plot(solution_simp_reg.t, solution_simp_reg.y[-1], label='simp_reg')
axs[1].plot(solution_z.t, solution_z.y[-1], label='Z')
axs[1].set_ylim(-0.3, 1.3)
axs[1].set_xlabel('Zeit [t]')
axs[1].set_ylabel('Konzentrationen')
axs[1].set_xticks([])
axs[1].set_yticks(np.arange(0, 1.1, 1))
axs[1].legend()

plt.tight_layout()
plt.show()


