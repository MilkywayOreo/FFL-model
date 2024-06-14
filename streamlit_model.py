import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")


#st.title("Feed Forward Loop")




st.latex(r"\frac{dY}{dt} = B_y + \beta_y  \left(\frac{ (\overset{*}X = S_x)/K_{xy}}{1 + \overset{*}X/K_{xy}}\right)^H - \alpha_y \, Y(t) \qquad \qquad  \frac{dZ}{dt} = B_z + \beta_z \left(\frac{\overset{*}X/K_{xz}}{1 + \overset{*}X/K_{xz}}\right)^H \left(\frac{(\overset{*}Y  = S_y \cdot Y(t))/K_{yz}}{1 + \overset{*}Y/K_{yz}}\right)^H - \alpha_z \, Z(t)")

col1, col2 = st.columns([2, 4])

with col1:
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        Kxz = st.slider("$\Large K_{xz}$", 0.0, 5.0, 0.1)
        Kxy = st.slider("$\Large K_{xy}$ - Equilibrium constant", 0.0, 5.0, 0.1)
        Kyz = st.slider("$\Large K_{yz}$", 0.0, 5.0, 0.5)
        alphay = st.slider("$\Large \\alpha_y$", 0.1, 5.0, 1.0)
        alphaz = st.slider("$\Large \\alpha_z$", 0.1, 5.0, 1.0)
        betay = st.slider("$\Large \\beta_y$", 0.1, 5.0, 1.0)
        betaz = st.slider("$\Large \\beta_z$", 0.1, 5.0, 1.0)
        H = st.slider("$\Large H$ - Hill coefficient", 0.1, 4.0, 2.0)
        By = st.slider("$\Large B_y$ - Basal expression ", 0.0, 1.0, 0.0)
        Bz = st.slider("$\Large B_z$", 0.0, 1.0, 0.0)
        Sx = st.slider("$\Large S_x$", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        Sy = st.slider("$\Large S_y$", 0.0, 10.0, 1.0)
        t_regler = st.slider("$\Large t$ - time", 0.0, 30.0, 10.0)
        tx = st.slider("$\Large t_{S_x}$", 0.0, 30.00, 7.00)
        ty = st.slider("$\Large t_{S_y}$", 0.0, 20.00, 20.00)

        

        
    with subcol2:
        AND_button = st.checkbox('AND GATE')
        OR_button = st.checkbox('OR GATE')
        C1_button = st.checkbox('C type 1')
        C2_button = st.checkbox('C type 2')
        C3_button = st.checkbox('C type 3')
        C4_button = st.checkbox('C type 4')
        I1_button = st.checkbox('I type 1')
        I2_button = st.checkbox('I type 2')
        I3_button = st.checkbox('I type 3')
        I4_button = st.checkbox('I type 4')


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

    def G(x_star, Kxz, y_star, Kyz, H):
        if AND_button and C1_button:
            return f_activator(x_star, Kxz, H) * f_activator(y_star, Kyz, H)
        elif AND_button and C2_button:
            return f_repressor(x_star, Kxz, H) * f_activator(y_star, Kyz, H)
        elif AND_button and C3_button:
            return f_repressor(x_star, Kxz, H) * f_repressor(y_star, Kyz, H)                     
        elif AND_button and C4_button:
            return f_activator(x_star, Kxz, H) * f_repressor(y_star, Kyz, H)

        elif AND_button and I1_button:
            return f_activator(x_star, Kxz, H) * f_repressor(y_star, Kyz, H) 
        elif AND_button and I2_button:
            return f_repressor(x_star, Kxz, H) * f_repressor(y_star, Kyz, H)
        elif AND_button and I3_button:
            return f_repressor(x_star, Kxz, H) * f_activator(y_star, Kyz, H)
        elif AND_button and I4_button:
            return f_activator(x_star, Kxz, H) * f_activator(y_star, Kyz, H) 

        elif OR_button and C1_button:
            return fc_activator(x_star, Kxz, Kyz, y_star, H) + fc_activator(y_star, Kyz, Kxz, x_star, H)
        elif OR_button and C2_button:
            return fc_repressor(x_star, Kxz, Kyz, y_star, H) + fc_activator(y_star, Kyz, Kxz, x_star, H)
        elif OR_button and C3_button:
            return fc_repressor(x_star, Kxz, Kyz, y_star, H) + fc_repressor(y_star, Kyz, Kxz, x_star, H)                     
        elif OR_button and C4_button:
            return fc_activator(x_star, Kxz, Kyz, y_star, H) + fc_repressor(y_star, Kyz, Kxz, x_star, H)  

        else:
            return 0

    def f(x_star, Kxy, H):
        if C1_button:
            return f_activator(x_star, Kxy, H)
        elif C2_button:
            return f_repressor(x_star, Kxy, H)
        elif C3_button:
            return f_activator(x_star, Kxy, H)                        
        elif C4_button:
            return f_repressor(x_star, Kxy, H)

        elif I1_button:
            return f_activator(x_star, Kxy, H)
        elif I2_button:
            return f_repressor(x_star, Kxy, H)                      
        elif I3_button:
            return f_activator(x_star, Kxy, H)
        elif I4_button:
            return f_repressor(x_star, Kxy, H)

        else:
            return 0



    def ODE_Y(t, initial_values, By, betay, Kxy, H, alphay):
        x, y, z = initial_values  # unpack x,y,z
        x_star = Sx(t)
        dydt = By + betay * f(x_star, Kxy, H) - alphay * y
        return [dydt]

    def ODE_Z(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
        x, y, z = initial_values
        x_star = Sx(t)
        y_star = Sy(t) * np.interp(t, solution_y.t, solution_y.y[-1]) # make y continuous    
        dzdt = Bz + betaz * G(x_star, Kxz, y_star, Kyz, H) - alphaz * z
        return [dzdt]

    def ODE_Z_simple_reg(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
        x, y, z = initial_values
        x_star = Sx(t)
        y_star = Sy(t)
        dzdt = Bz + betaz * G(x_star, Kxz, y_star, Kyz, H) - alphaz * z
        return [dzdt]
 
    tx_end = tx
    Sx_regler = float(Sx)
    def Sx(t):
        if 1 < t < tx_end:
            return Sx_regler
        else:
            return 0

    ty_end = ty
    Sy_regler = float(Sy)
    def Sy(t):
        if 1 < t < ty_end:
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
    ax[1].axvline(x=tx_end + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
    ax[1].axvline(x=1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(x=tx_end, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(x=tx_end + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
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
