import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")


#st.title("Feed Forward Loop")

st.latex(r"\frac{dY}{dt} = B_y + \beta_y  \left(\frac{ (\overset{*}X = S_x)/K_{xy}}{1 + \overset{*}X/K_{xy}}\right)^H - \alpha_y \, Y(t) \qquad \qquad  \frac{dZ}{dt} = B_z + \beta_z \left(\frac{\overset{*}X/K_{xz}}{1 + \overset{*}X/K_{xz}}\right)^H \left(\frac{(\overset{*}Y  = Y(t)\cdot S_y)/K_{yz}}{1 + \overset{*}Y/K_{yz}}\right)^H - \alpha_z \, Z(t)")
                  

col1, col2 = st.columns([2, 4])

with col1:
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        Kxy = st.slider("$K_{xy}$ - Equilibrium constant", 0.01, 1.0, 0.1)
        Kxz = st.slider("$K_{xz}$", 0.01, 1.0, 0.1)
        Kyz = st.slider("$K_{yz}$", 0.01, 1.0, 0.5)
        By = st.slider("$B_y$ - Basal expression ", 0.0, 1.0, 0.0)
        Bz = st.slider("$B_z$", 0.0, 1.0, 0.0)
        Sx = st.slider("$S_x$", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        t_regler = st.slider("t - time", 0.0, 20.0, 10.0)

        
    with subcol2:
        betay = st.slider("$\\beta_y$", 0.1, 5.0, 1.0)
        betaz = st.slider("$\\beta_z$", 0.1, 5.0, 1.0)
        alphay = st.slider("$\\alpha_y$", 0.1, 5.0, 1.0)
        alphaz = st.slider("$\\alpha_z$", 0.1, 5.0, 1.0)
        H = st.slider("H - Hill coefficient ", 0.1, 4.0, 2.0)
        Sy = st.slider("$S_y$", 0.0, 10.0, 1.0)

    

with col2:
    # Hill Functions
    def f_activator(u, K, H):
        return (u/K)**H / (1 + (u/K)**H)
    def f_repressor(u, K, H):
        return 1 / (1 + (u/K)**H) 
    def fc_activator(u, Ku, Kv, v, H):
        return (u / Ku)**H / (1 + (u/Ku)**H + (v/Kv)**H)
    def fc_repressor(u, Ku, Kv, v, H):
        return 1 / (1 + (u/Ku)**H + (v/Kv)**H)    

    def ODE_Y(t, initial_values, By, betay, Kxy, H, alphay):
        x, y, z = initial_values  # unpack x,y,z
        x_star = Sx(t)
        dydt = By + betay * f_activator(x_star, Kxy, H) - alphay * y
        return [dydt]

    def ODE_Z(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
        x, y, z = initial_values
        x_star = Sx(t)
        y_star = Sy(t) * np.interp(t, solution_y.t, solution_y.y[-1]) # make y continuous    
        dzdt = Bz + betaz * f_activator(x_star, Kxz, H) * f_activator(y_star, Kyz, H) - alphaz * z
        return [dzdt]

    def ODE_Z_simple_reg(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
        x, y, z = initial_values
        x_star = Sx(t)
        y_star = Sy(t)
        dzdt = Bz + betaz * f_activator(x_star, Kxz, H) * f_activator(y_star, Kyz, H) - alphaz * z
        return [dzdt]

    Sx_regler = float(Sx)
    def Sx(t):
        if 1 < t < 7:
            return Sx_regler
        else:
            return 0

    Sy_regler = float(Sy)
    def Sy(t):
        if 1 < t < 7:
            return Sy_regler
        else:
            return 0            
    
    t_end = t_regler
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 1000)

    initial_values = [Sx_regler, By, Bz]

    solution_y = solve_ivp(ODE_Y, t_span, initial_values, t_eval=t_eval, method='Radau', args=(By, betay, Kxy, H, alphay))
    solution_z = solve_ivp(ODE_Z, t_span, initial_values, t_eval=solution_y.t, method='Radau', args=(Bz, betaz, Kxz, Kyz, H, alphaz))
    solution_z_simple_reg = solve_ivp(ODE_Z_simple_reg, t_span, initial_values, t_eval=t_eval, method='Radau', args=(Bz, betaz, Kxz, Kyz, H, alphaz))


    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [0.3, 0.7]})

    Sx_vals = [Sx(t) for t in t_eval]
    Sy_vals = [Sy(t) for t in t_eval]
    ax[0].plot(t_eval, Sx_vals, label='$S_x$', color='purple')
    ax[0].plot(t_eval, Sy_vals, label='$S_y$')
    #ax[0].set_ylabel('$S_x$', rotation=360, fontsize="15")
    ax[0].set_xticks([])
    ax[0].set_yticks(np.arange(0, Sx_regler+0.1, 1))
    ax[0].set_yticks(np.arange(0, Sy_regler+0.1, 1))
    ax[0].legend(loc="center left", fontsize='16', frameon=False)
    ax[1].axvline(x=7 + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
    ax[1].axvline(x=1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(x=7, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(x=7 + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
    ax[1].axhline(y=np.max(solution_z.y[-1])/2.0, color='k', linestyle='--', linewidth=1)
    ax[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax[1].axhline(y=1, color='k', linestyle='--', linewidth=1)
    if st.checkbox('Y(t) anzeigen'):
        ax[1].plot(solution_y.t, solution_y.y[-1], label='Y(t)', color= 'green')
    ax[1].plot(solution_z_simple_reg.t, solution_z_simple_reg.y[-1], label='$Z(t)_{simple}$')
    ax[1].plot(solution_z.t, solution_z.y[-1], label='$Z(t)_{FFL}$')
    ax[1].set_ylim(-0.3, 1.3)
    ax[1].set_xlabel('time [t]', fontsize="15")
    #ax[1].set_ylabel('Z', rotation=360, fontsize="15")
    ax[1].set_xticks([7 + np.log(2)/alphaz])
    ax[1].set_xticklabels([r"$\tau = \frac{\ln(2)}{\alpha_z}$"], fontsize=15)
    ax[1].set_yticks(np.arange(0, 1.1, 1))
    ax[1].legend(loc='center left', bbox_to_anchor=(-0.01, 0.65), fontsize='15', frameon=False)

    st.pyplot(fig)
