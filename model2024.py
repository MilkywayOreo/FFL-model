import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# parameter
H = 2 # Hill coeficient
b_y, b_z = 1
a_y, a_z = 1 # a_deg + a_dil
B_y, B_z = 0 # Basal expression of y,z

# chemical equilibrium constants
K_xy 
K_xz
K_yz

# simple Activator
def activator(u, K, H):
    return (u/K)**H / (1 + (u/K)**H) 
# simple Repressor
def repressor(u, K, H):
	return 1 / (1 + (u/K)**H) 

# competive Activator G_z
def fc_act(u, Ku, Kv, v, H):
    return (u / Ku)**H / (1 + (u/Ku)**H + (v/Kv)**H)
# competive Repressor G_z
def fc_rep(u, Ku, Kv, v, H):
    return 1 / (1 + (u/Ku)**H + (v/Kv)**H)

def dYdt():
    X_star_effect = Sx * X_star  # Ein- und Ausschalten von Sx
    if input("X -> Y: Aktivator A / Repressor R: ") == A:
	    return B_y + b_y * f(X_star_effect, Kxy, H) - ay * Y
	else:
		return 

def dZdt():
	if input("AND Gate oder OR Gate: ") == AND:    
		return B_z
	else:
		return:		



def dZdt(t, Z, X_star, Kxz, Y_star, Kyz, az, Bz, bz, H, Sx):
    X_star_effect = Sx * X_star  # Ein- und Ausschalten von Sx
    return Bz + bz * fc(X_star_effect, Kxz, Kyz, Y_star, H) - az * Z

def system(t, variables, Kxy, Kxz, Kyz, ay, By, by, az, Bz, bz, H, Sx):
    X_star, Y, Z = variables
    dXdt = 0  # Angenommen, X wird konstitutiv ausgedrückt
    dYdt_val = dYdt(t, Y, X_star, Kxy, ay, By, by, H, Sx)
    dZdt_val = dZdt(t, Z, X_star, Kxz, Y, Kyz, az, Bz, bz, H, Sx)
    return [dXdt, dYdt_val, dZdt_val]    
