import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

Y_check = st.checkbox('Y(t) anzeigen')
#st.title("Feed Forward Loop")

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


button = st.columns(10)
AND_button = button[0].checkbox('AND GATE', value=True)
OR_button = button[1].checkbox('OR GATE')
C1_button = button[2].checkbox('C type 1', value=True)
C2_button = button[3].checkbox('C type 2', value=True)
C3_button = button[4].checkbox('C type 3', value=True)
C4_button = button[5].checkbox('C type 4', value=True)
I1_button = button[6].checkbox('I type 1', value=True)
I2_button = button[7].checkbox('I type 2', value=True)
I3_button = button[8].checkbox('I type 3', value=True)
I4_button = button[9].checkbox('I type 4', value=True)


eq, dgl = st.columns([1,7])
if OR_button and C1_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl:    
        st.latex(dydt_eq_C1_OR + r"\qquad \qquad"+ dzdt_eq_C1_OR)
elif OR_button and C2_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C2_OR + r"\qquad \qquad"+ dzdt_eq_C2_OR)
elif OR_button and C3_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C3_OR + r"\qquad \qquad"+ dzdt_eq_C2_OR)
elif OR_button and C4_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C4_OR + r"\qquad \qquad"+ dzdt_eq_C3_OR)

elif AND_button and C1_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C1 + r"\qquad \qquad"+ dzdt_eq_C1)
elif AND_button and C2_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C2 + r"\qquad \qquad"+ dzdt_eq_C2)
elif AND_button and C3_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C3 + r"\qquad \qquad"+ dzdt_eq_C3)
elif AND_button and C4_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_C4 + r"\qquad \qquad"+ dzdt_eq_C4)

elif AND_button and I1_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_I1 + r"\qquad \qquad"+ dzdt_eq_I1)
elif AND_button and I2_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(dydt_eq_I2 + r"\qquad \qquad"+ dzdt_eq_I2)
elif AND_button and I3_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(eq_Ystar + dydt_eq_I3 + r"\qquad \qquad"+ dzdt_eq_I3)
elif AND_button and I4_button:
    with eq:
        st.latex(eq_Xstar + r"\\" + eq_Ystar)
    with dgl: 
        st.latex(eq_Ystar + dydt_eq_I4 + r"\qquad \qquad"+ dzdt_eq_I4)



params, PLOT = st.columns([2, 3])
with params:
    slider1, slider2 = st.columns(2)
    
    with slider1:
        Kxy = st.slider("$\Large K_{xy}$ - Equilibrium constant", 0.01, 5.0, 0.1)
        Kxz = st.slider("$\Large K_{xz}$", 0.01, 5.0, 0.1)
        Kyz = st.slider("$\Large K_{yz}$", 0.01, 5.0, 0.5)
        H = st.slider("$\Large H$ - Hill coefficient", 0.1, 200.0, 2.0)
        Sx = st.slider("$\Large S_x$", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        Sy = st.slider("$\Large S_y$", 0.01, 10.0, 1.0)
        tx = st.slider("$\Large t_{S_x}$", 0.0, 30.00, 7.00)
        t_regler = st.slider("$\Large t$ - time", 0.0, 30.0, 10.0)
          
    with slider2:
        alphay = st.slider("$\Large \\alpha_y$", 0.1, 10.0, 1.0)
        alphaz = st.slider("$\Large \\alpha_z$", 0.1, 10.0, 1.0)
        betay = st.slider("$\Large \\beta_y$", 0.1, 10.0, 1.0)
        betaz = st.slider("$\Large \\beta_z$", 0.1, 10.0, 1.0)
        By = st.slider("$\Large B_y$ - Basal expression ", 0.0, 10.0, 0.0)
        Bz = st.slider("$\Large B_z$", 0.0, 10.0, 0.0)
        ty = st.slider("$\Large t_{S_y}$", 0.0, 20.00, 20.00)


        

with PLOT:
    # Hill Functions
    def f_activator(u, K, H):
        return (u/K)**H / (1 + (u/K)**H)
    def f_repressor(u, K, H):
        return 1 / (1 + (u/K)**H) 
    def fc_activator(u, Ku, Kv, v, H):
        return (u / Ku)**H / (1 + (u/Ku)**H + (v/Kv)**H)
    def fc_repressor(u, Ku, Kv, v, H):
        return 1 / (1 + (u/Ku)**H + (v/Kv)**H)    

    def f(x_star, Kxy, H):
        if C1_button:
            return f_activator(x_star, Kxy, H)
        if C2_button:
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

    def G(x_star, Kxz, y_star, Kyz, H):
        if AND_button and C1_button:
            return f_activator(x_star, Kxz, H) * f_activator(y_star, Kyz, H)
        if AND_button and C2_button:
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
    z_max = np.max(solution_z_simple_reg.y[-1])


    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [0.15, 0.85]})


    Sx_vals = [Sx(t) for t in t_eval]
    Sy_vals = [Sy(t) for t in t_eval]
    ax[0].plot(t_eval, Sx_vals, label='$S_x$', color='orange')
    ax[0].plot(t_eval, Sy_vals, label='$S_y$', color='DarkOrchid')
    #ax[0].set_ylabel('$S_x$', rotation=360, fontsize="15")
    ax[0].set_xticks([])
    ax[0].set_yticks(np.arange(0, Sx_regler+0.1, 1))
    ax[0].set_yticks(np.arange(0, Sy_regler+0.1, 1))
    ax[0].legend(fontsize='14', frameon=False)

    ax[1].axvline(x=tx_end + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
    ax[1].axvline(x=1, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(x=tx_end, color='k', linestyle='--', linewidth=1)
    if 2.0 * z_max != 0:
        ax[1].axvline(x=tx_end + np.log(2)/alphaz, color='k', linestyle='--', linewidth=1) 
        ax[1].axhline(y=np.max(solution_z.y[-1])/(2.0*z_max), color='k', linestyle='--', linewidth=1)
        ax[1].text(x=0, y=np.max(solution_z.y[-1]) / (2.0 * z_max), s='1/2 max_FFL', fontsize=12, va='center', ha='left', backgroundcolor='w')
        ax[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax[1].axhline(y=np.max(solution_z_simple_reg.y[-1]/z_max), color='k', linestyle='--', linewidth=1)

    if Y_check:
        ax[1].plot(solution_y.t, solution_y.y[-1], label='Y(t)', color= 'green')
    ax[1].plot(solution_z_simple_reg.t, solution_z_simple_reg.y[-1]/z_max, label='$Z(t)_{simple}$')
    ax[1].plot(solution_z.t, solution_z.y[-1]/z_max, label='$Z(t)_{FFL}$')
    ax[1].set_ylim(-0.3, 1.8)
    ax[1].set_xlabel('time [t]', fontsize="15")
    ax[1].set_ylabel('Z', rotation=360, fontsize="15")
    ax[1].set_xticks([tx_end + np.log(2)/alphaz])
    ax[1].set_xticklabels([r"$\tau = \frac{\ln(2)}{\alpha_z}$"], fontsize=15)
    ax[1].set_yticks(np.arange(0, 1.1, 1))
    ax[1].legend(fontsize='13', frameon=False)

    st.pyplot(fig)



