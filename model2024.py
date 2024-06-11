import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# parameters
H = 2  # Hill coefficient
betay = 1
betaz = 1
alphay = 1
alphaz = 1  # a_deg + a_dil
By = 0  # Basal expression of y, z
Bz = 0
Kxy = 1  # equilibrium constants
Kxz = 1
Kyz = 1

def f_activator(u, K, H):
    return (u/K)**H / (1 + (u/K)**H)

# Zeitschalter Sx
def Sx(t):
    if 3 < t < 10:
        return 1
    else:
        return 0

def ODE_Y(t, initial_values, By, betay, Kxy, H, alphay):
    x, y, z = initial_values  # entpacken von x,y,z
    x = Sx(t)
    dydt = By + betay * f_activator(x, Kxy, H) - alphay * y
    return [dydt]

def ODE_Z(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
    x, y, z = initial_values
    x = Sx(t)
    y = np.interp(t, solution_y.t, solution_y.y[0])
    dzdt = Bz + betaz * f_activator(x, Kxz, H) * f_activator(y, Kyz, H) - alphaz * z
    return [dzdt]

def simp_reg(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
    x, y, z = initial_values
    x = Sx(t)
    y = 1
    dzdt = Bz + betaz * f_activator(x, Kxz, H) * f_activator(y, Kyz, H) - alphaz * z
    return [dzdt]


initial_values = [1, By, Bz]

# Zeitbereich für die Lösung
t_span = (0, 15)
t_eval = np.linspace(0, 15, 100)

# ODEs lösen
solution_y = solve_ivp(ODE_Y, t_span, initial_values, t_eval=t_eval, args=(By, betay, Kxy, H, alphay))
solution_z = solve_ivp(ODE_Z, t_span, initial_values, t_eval=t_eval, args=(Bz, betaz, Kxz, Kyz, H, alphaz))
solution_simp_reg = solve_ivp(simp_reg, t_span, initial_values, t_eval=t_eval, args=(Bz, betaz, Kxz, Kyz, H, alphaz))


# Plotten der Ergebnisse
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

Sx = [Sx(t) for t in t_eval]
axs[0].plot(t_eval, Sx, label='Sx', color='red')
axs[0].set_ylabel('Sx')
axs[0].legend()

axs[1].plot(solution_y.t, solution_y.y[0], label='Y')
axs[1].plot(solution_z.t, solution_z.y[0], label='Z')
axs[1].plot(solution_simp_reg.t, solution_simp_reg.y[0], label='simp_reg')
axs[1].set_xlabel('Zeit [t]')
axs[1].set_ylabel('Konzentrationen')
axs[1].legend()

plt.tight_layout()
plt.show()
