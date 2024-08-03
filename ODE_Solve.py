import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# define the differential equations

def NI(t, Z, A, B, C, d, e):
    Nn, Im, I = Z

    dNn_dt = A*(1-Nn)/(e+I) - B*I*Nn/(d+Nn)
    dIm_dt = Nn*Nn - Im
    dI_dt = Im - C*I*(1-Nn)/(e+I)

    return [dNn_dt, dIm_dt, dI_dt]


# Initial conditions for solving differential equations and parameters
y0 = [1.0, 0.0, 0.0]
t_span = (0, 1000)
t_eval = np.arange(0, 1000.01, 0.01)
A, B, C, d, e = 0.007, 954.5, 0.035, 0.029, 2e-5


def solve_ode(A, B, C, d, e):
    sol = solve_ivp(NI, t_span, y0, t_eval=t_eval, args=(A, B, C, d, e,), dense_output=False, method='LSODA')
   
    return sol.t[8000:12000]/0.018, sol.y[0,8000:12000]    # time 4444min to 6666min: steady oscillations



tt, Nn  = solve_ode(A, B, C, d, e)

# find maxima and minima of the oscillations

maxima, _ = find_peaks(Nn, prominence = 0.1)

inverted_array = -Nn
minima, _ = find_peaks(inverted_array, prominence = 0.1)

n_extrema = min(len(maxima), len(minima))    # get the minima of the number of maxima and minima to plot and find 

    
# plot the oscillations with the extremas
plt.figure()
plt.plot(tt, Nn, label='Data')
plt.plot(tt[maxima[:n_extrema]], Nn[maxima[:n_extrema]], "X", label='Maxima', color = 'r')
plt.plot(tt[minima[:n_extrema]], Nn[minima[:n_extrema]], "o", label='Minima', color = 'k')
plt.title(f'A = {A}, C = {C}')
plt.legend()
plt.show()

# mean maxima and minima 
Mean_Nn_max = np.mean(Nn[maxima[:n_extrema]])
Mean_Nn_min = np.mean(Nn[minima[:n_extrema]])

# time period
TimePeriod = np.diff(tt[maxima])
Mean_TimePeriod = np.mean(TimePeriod)
#Std_TimePeriod = np.std(TimePeriod)

# amplitude
Amplitude = np.abs(Nn[maxima[:n_extrema]]-Nn[minima[:n_extrema]])
Mean_Amplitude = np.mean(Amplitude)
# Std_Amplitude = np.std(Amplitude)
    
# Z value
Z = (Mean_Nn_max-Mean_Nn_min)/np.mean(Nn)

print("Mean time period =", Mean_TimePeriod, "\nMean amplitude =", Mean_Amplitude, "\nZ =",Z)    
 


