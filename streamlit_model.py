import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st


def interactivModel(eq_Xstar, eq_Ystar, dydt_eq_C1_OR, dydt_eq_C2_OR, dydt_eq_C3_OR, dydt_eq_C4_OR, dydt_eq_C1, dydt_eq_C2, dydt_eq_C3, dydt_eq_C4, dydt_eq_I1, dydt_eq_I2, dydt_eq_I3, dydt_eq_I4, dzdt_eq_C1_OR, dzdt_eq_C2_OR, dzdt_eq_C3_OR, dzdt_eq_C4_OR, dzdt_eq_C1, dzdt_eq_C2, dzdt_eq_C3, dzdt_eq_C4, dzdt_eq_I1, dzdt_eq_I2, dzdt_eq_I3, dzdt_eq_I4):
    
    # CHECKBOXs
    normalize = st.checkbox(r'Normalisieren bzgl. $Z(t)_{simple}$')
    AND_C1_simple = st.checkbox('AND C1 simple')    
    Y_check = st.checkbox('Y(t)')

    time = st.columns(10)
    tx_slider = time[0].slider("$\Large t_{S_x}$", 0.0, 30.00, 7.00)
    ty_slider = time[1].slider("$\Large t_{S_y}$", 0.0, 20.00, 20.00)
    t_slider = time[2].slider("$\Large t$ - time", 0.0, 30.0, 10.0)
    
    button = st.columns(12)
    AND_button = button[0].checkbox('AND', value=True)
    OR_button = button[1].checkbox('OR')
    if OR_button:
        AND_button = False        
    C1_button = button[3].checkbox('C1', value=True)
    C2_button = button[4].checkbox('C2')
    C3_button = button[5].checkbox('C3')
    C4_button = button[6].checkbox('C4', value=True)
    I1_button = button[8].checkbox('I1', value=True)
    I2_button = button[9].checkbox('I2')
    I3_button = button[10].checkbox('I3')
    I4_button = button[11].checkbox('I4', value=True)

    regler = st.columns(12)
    Sx_slider = regler[0].slider("$\Large S_x$", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    Sy_slider = regler[1].slider("$\Large S_y$", 0.01, 10.0, 1.0)

    By = regler[2].slider("$\Large B_y$", 0.0, 1.0, 0.0)
    Bz = regler[3].slider("$\Large B_z$", 0.0, 1.0, 0.0)
    H = regler[4].slider("$\Large H$", 0.1, 200.0, 2.0)

    alphay = regler[5].slider("$\Large \\alpha_y$", 0.1, 10.0, 1.0)
    alphaz = regler[6].slider("$\Large \\alpha_z$", 0.1, 10.0, 1.0)

    betay = regler[7].slider("$\Large \\beta_y$", 0.1, 10.0, 1.0)
    betaz = regler[8].slider("$\Large \\beta_z$", 0.1, 10.0, 1.0)

    Kxy = regler[9].slider("$\Large K_{xy}$", 0.01, 5.0, 0.1)
    Kxz = regler[10].slider("$\Large K_{xz}$", 0.01, 5.0, 0.1)
    Kyz = regler[11].slider("$\Large K_{yz}$", 0.01, 5.0, 0.5)



    # Gleichung mit Checkbox auswählen
    dgl, plot = st.columns([10,9])        
    with dgl:

        if OR_button and C1_button:
            st.latex(eq_Xstar +r"\\" + eq_Ystar)
            st.latex(dydt_eq_C1_OR +r"\\") 
            st.latex(dzdt_eq_C1_OR)
        elif OR_button and C2_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C2_OR + r"\\") 
            st.latex(dzdt_eq_C2_OR)
        elif OR_button and C3_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C3_OR + r"\\") 
            st.latex(dzdt_eq_C3_OR)
        elif OR_button and C4_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C4_OR + r"\\") 
            st.latex(dzdt_eq_C4_OR)
        elif AND_button and C1_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C1 + r"\\") 
            st.latex(dzdt_eq_C1)
        elif AND_button and C2_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C2 + r"\\") 
            st.latex(dzdt_eq_C2)
        elif AND_button and C3_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C3 + r"\\") 
            st.latex(dzdt_eq_C3)
        elif AND_button and C4_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_C4 + r"\\") 
            st.latex(dzdt_eq_C4)
        elif AND_button and I1_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_I1 + r"\\") 
            st.latex(dzdt_eq_I1)
        elif AND_button and I2_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_I2 + r"\\") 
            st.latex(dzdt_eq_I2)
        elif AND_button and I3_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_I3 + r"\\") 
            st.latex(dzdt_eq_I3)
        elif AND_button and I4_button:
            st.latex(eq_Xstar + r"\\" + eq_Ystar)
            st.latex(dydt_eq_I4 + r"\\") 
            st.latex(dzdt_eq_I4)








    # Plot
    with plot:
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
            dydt = By + betay * f_activator(x_star, Kxy, H) - alphay * y
            return [dydt]


        def ODE_Model(t, initial_values, By, Bz, betay, betaz, Kxz, Kyz, H, alphay, alphaz):
            x, y, z = initial_values
            x_star = Sx(t)
            y_star = Sy(t) * y

            dxdt = 0
            dydt = By + betay * f(x_star, Kxy, H) - alphay * y
            dzdt = Bz + betaz * G(x_star, Kxz, y_star, Kyz, H) - alphaz * z
            # dzdt = Bz + betaz * (fc_activator(x_star, Kxz, Kyz, y_star, H) + fc_activator(y_star, Kyz, Kxz, x_star, H)) - alphaz * z
            return [dxdt, dydt, dzdt]

        def ODE_Z_simple_reg(t, initial_values, Bz, betaz, Kxz, Kyz, H, alphaz):
            x, y, z = initial_values
            x_star = Sx(t)
            y_star = Sy(t)
            #Kxz = 1
            if AND_C1_simple:
                dzdt = Bz + betaz * (f_activator(x_star, Kxz, H) * f_activator(y_star, Kyz, H)) - alphaz * z
            else:
                dzdt = Bz + betaz * G(x_star, Kxz, y_star, Kyz, H) - alphaz * z

            return [dzdt]
     
        tx_end = float(tx_slider)
        Sx_val = float(Sx_slider)
        def Sx(t):
            if 1 < t < tx_end:
                return Sx_val
            else:
                return 0

        ty_end = float(ty_slider)
        Sy_val = float(Sy_slider)
        def Sy(t):
            if 1 < t < ty_end:
                return Sy_val
            else:
                return 0            
        
        t_end = float(t_slider) 
        t_span = (0, t_end)
        t_eval = np.linspace(0, t_end, 1000)

        initial_values = [Sx_val, 0, 0]

        solution_y = solve_ivp(ODE_Y, t_span, initial_values, t_eval=t_eval, method = 'Radau', args=(By, betay, Kxy, H, alphay)) # RK45, RK23, DOP853, Radau...
        solution_z = solve_ivp(ODE_Model, t_span, initial_values, t_eval=t_eval, method='Radau', args=(By, Bz, betay, betaz, Kxz, Kyz, H, alphay, alphaz))
        solution_z_simple_reg = solve_ivp(ODE_Z_simple_reg, t_span, initial_values, t_eval=t_eval, method='Radau', args=(Bz, betaz, Kxz, Kyz, H, alphaz))

        decay_start = np.searchsorted(solution_z.t, tx_end, side='left')
        z_max_FFL = np.max(solution_z.y[-1][decay_start:])
        z_max_simple = np.max(solution_z_simple_reg.y[-1])


        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [0.15, 0.85]})

        Sx_vals = [Sx(t) for t in t_eval]
        Sy_vals = [Sy(t) for t in t_eval]
        ax[0].plot(t_eval, Sx_vals, label='$S_x$', color='orange')
        ax[0].plot(t_eval, Sy_vals, label='$S_y$', color='DarkOrchid')
        #ax[0].set_ylabel('$S_x$', rotation=360, fontsize="15")
        ax[0].set_xticks([])
        ax[0].set_yticks(np.arange(0, Sx_slider+0.1, 1))
        ax[0].set_yticks(np.arange(0, Sy_slider+0.1, 1))
        ax[0].legend(fontsize='13', frameon=False)


        if Y_check:
            if normalize:
                ax[1].plot(solution_y.t, solution_y.y[-1]/z_max_simple, label='Y(t)', color= 'green')
            else:     
                ax[1].plot(solution_y.t, solution_y.y[-1], label='Y(t)', color= 'green')

        if normalize and z_max_simple != 0:
            ax[1].plot(solution_z_simple_reg.t, solution_z_simple_reg.y[-1]/z_max_simple, label='$Z(t)_{simple}$')
            ax[1].plot(solution_z.t, solution_z.y[-1]/z_max_simple, label='$Z(t)_{FFL}$')
            ax[1].axhline(y=z_max_FFL/(2.0 * z_max_simple), color='k', linestyle='--', linewidth=1)
            ax[1].text(x=0, y=z_max_FFL/(2.0 * z_max_simple), s='1/2 max_FFL', fontsize=12, va='center', ha='right', backgroundcolor='w')
        else:
            ax[1].plot(solution_z_simple_reg.t, solution_z_simple_reg.y[-1], label='$Z(t)_{simple}$')
            ax[1].plot(solution_z.t, solution_z.y[-1], label='$Z(t)_{FFL}$')
            ax[1].axhline(y=z_max_FFL/2.0, color='k', linestyle='--', linewidth=1)
            #ax[1].text(x=0, y=z_max_FFL/2.0, s='1/2 max_FFL', fontsize=9, va='center', ha='left', backgroundcolor='w')

        if z_max_simple != 0:
            ax[1].axhline(y=np.max(solution_z_simple_reg.y[-1])/z_max_simple, color='gray', linestyle='--', linewidth=1)

        ax[1].axvline(x=tx_end + np.log(2)/alphaz, color='gray', linestyle='--', linewidth=1) 
        ax[1].axvline(x=1, color='gray', linestyle='--', linewidth=1)
        ax[1].axvline(x=tx_end, color='gray', linestyle='--', linewidth=1)
        ax[1].axvline(x=tx_end + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
        ax[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)

        ax[1].set_ylim(-0.3, 1.5)
        ax[1].set_xlabel('time [t]', fontsize="15")
        #ax[1].set_ylabel('Z', rotation=360, fontsize="15")
        ax[1].set_xticks([tx_end + np.log(2)/alphaz])
        ax[1].set_xticklabels([r"$\tau = \frac{\ln(2)}{\alpha_z}$"], fontsize=15)
        ax[1].set_yticks([0,  1])
        ax[1].legend(fontsize='13', frameon=False)

        st.pyplot(fig)


    Gleichungen_button = st.checkbox("alle Gleichungen")
    name, gleichung = st.columns([1,7])
    if Gleichungen_button:
        with name: 
            st.markdown("#")
            st.markdown("#")
            st.markdown("# AND")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("# AND")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("# OR")



        with gleichung: 
            st.markdown("#")
            st.markdown("#")

            st.latex("C1: \qquad" +dydt_eq_C1 + r"\qquad \qquad"+ dzdt_eq_C1)

            st.latex("C2: \qquad" +dydt_eq_C2 + r"\qquad \qquad"+ dzdt_eq_C2)

            st.latex("C3: \qquad" +dydt_eq_C3 + r"\qquad \qquad"+ dzdt_eq_C3)

            st.latex("C4: \qquad" +dydt_eq_C4 + r"\qquad \qquad"+ dzdt_eq_C4)

            st.markdown("---")

            st.latex("I1: \qquad" +dydt_eq_I1 + r"\qquad \qquad"+ dzdt_eq_I1)

            st.latex("I2: \qquad" +dydt_eq_I2 + r"\qquad \qquad"+ dzdt_eq_I2)

            st.latex("I3: \qquad" +dydt_eq_I3 + r"\qquad \qquad"+ dzdt_eq_I3)

            st.latex("I4: \qquad" +dydt_eq_I4 + r"\qquad \qquad"+ dzdt_eq_I4)

            st.markdown("---")

            st.latex("C1: \qquad" + dydt_eq_C1_OR + r"\qquad \qquad"+ dzdt_eq_C1_OR)

            st.latex("C2: \qquad" +dydt_eq_C2_OR + r"\qquad \qquad"+ dzdt_eq_C2_OR)

            st.latex("C3: \qquad" +dydt_eq_C3_OR + r"\qquad \qquad"+ dzdt_eq_C2_OR)

            st.latex("C4: \qquad" +dydt_eq_C4_OR + r"\qquad \qquad"+ dzdt_eq_C3_OR)



def setup():
    st.set_page_config(layout="wide")

    st.title("Feed Forward Loop")

    #dYdt
    dydt_eq_C1_OR = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xy}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"
    dydt_eq_C2_OR = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"
    dydt_eq_C3_OR = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xy}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"    
    dydt_eq_C4_OR = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"

    dydt_eq_C1 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xy}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t) "      
    dydt_eq_C2 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"
    dydt_eq_C3 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xy}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"     
    dydt_eq_C4 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"

    dydt_eq_I1 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xy}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"        
    dydt_eq_I2 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"        
    dydt_eq_I3 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xy}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"      
    dydt_eq_I4 = r"\frac{dY}{dt} = B_y + \beta_y \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xy}} \right)^H - \alpha_y \, Y(t)"        

    #dZdt
    dzdt_eq_C1_OR = r"\frac{dZ}{dt} = B_z + \beta_z \left[ \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xz}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz} + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H + \left( \frac{\textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz} + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \right] - \alpha_z \, Z(t)"
    dzdt_eq_C2_OR = r"\frac{dZ}{dt} = B_z + \beta_z \left[ \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz} + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H + \left( \frac{\textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz} + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \right] - \alpha_z \, Z(t)"
    dzdt_eq_C3_OR = r"\frac{dZ}{dt} = B_z + \beta_z \left[ \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz} + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H + \left( \frac{1}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz} + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \right] - \alpha_z \, Z(t)"
    dzdt_eq_C4_OR = r"\frac{dZ}{dt} = B_z + \beta_z \left[ \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xz}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz} + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H + \left( \frac{1}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz} + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \right] - \alpha_z \, Z(t)"

    dzdt_eq_C1 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xz}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{\textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"
    dzdt_eq_C2 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{\textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"
    dzdt_eq_C3 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{1}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"
    dzdt_eq_C4 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xz}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{1}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"

    dzdt_eq_I1 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xz}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{1}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"
    dzdt_eq_I2 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{1}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"
    dzdt_eq_I3 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{1}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{\textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"
    dzdt_eq_I4 = r"\frac{dZ}{dt} = B_z + \beta_z \left( \frac{\textcolor{orange}{\overset{*}{X}} / K_{xz}}{1 + \textcolor{orange}{\overset{*}{X}} / K_{xz}} \right)^H \left( \frac{\textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}}{1 + \textcolor{DarkOrchid}{\overset{*}{Y}} / K_{yz}} \right)^H - \alpha_z \, Z(t)"

    eq_Xstar = r"\textcolor{orange}{\overset{*}{X}} = S_x"
    eq_Ystar = r"\textcolor{DarkOrchid}{\overset{*}{Y}}= S_y \cdot Y(t)"

    interactivModel(eq_Xstar, eq_Ystar, dydt_eq_C1_OR, dydt_eq_C2_OR, dydt_eq_C3_OR, dydt_eq_C4_OR, dydt_eq_C1, dydt_eq_C2, dydt_eq_C3, dydt_eq_C4, dydt_eq_I1, dydt_eq_I2, dydt_eq_I3, dydt_eq_I4, dzdt_eq_C1_OR, dzdt_eq_C2_OR, dzdt_eq_C3_OR, dzdt_eq_C4_OR, dzdt_eq_C1, dzdt_eq_C2, dzdt_eq_C3, dzdt_eq_C4, dzdt_eq_I1, dzdt_eq_I2, dzdt_eq_I3, dzdt_eq_I4)


setup()    
