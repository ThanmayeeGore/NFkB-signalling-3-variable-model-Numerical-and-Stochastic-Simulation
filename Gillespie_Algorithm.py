import numpy as np
import matplotlib.pyplot as plt
import time as tt
import pandas as pd


def GA(mol_num, tfinal, k_Nin, alpha):

    stochiometry_matrix = np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]])
    
    t = 0
    time = [t]
    
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])    # propensities

    
    X = np.array([int(mol_num), 0, 0])    # initial condition
    x1 = [X[0]]   # N_n
    x2 = [X[1]]   # I_m
    x3 = [X[2]]   # I

    N_tot, K_I, k_Iin, K_N, k_t, k_tI, gamma_m = mol_num, 0.0185*mol_num, 0.018, 0.029*mol_num, 1.03/mol_num, 0.24, 0.018

    start = tt.time()    # keep track of time
    
    while t<tfinal:    
    
        a[0] = k_Nin*(N_tot - X[0])*(K_I/(K_I + X[2]))     # propensity for production of N_n 
        a[1] = k_Iin*X[2]*X[0]/(K_N+X[0])                  # propensity for degradation of N_n
        a[2] = k_t*X[0]*(X[0]-1)                           # propensity for production of I_m
        a[3] = gamma_m*X[1]                                # propensity for degradation of I_m
        a[4] = k_tI*X[1]                                   # propensity for production of I
        a[5] = 0.5*alpha*(N_tot-X[0])*X[2]/(K_I+X[2])      # propensity for degradation of I
        
        asum = np.sum(a)                              

        if asum == 0:
            print('break')
            break
    
        r = np.random.rand(2)
    
        j = np.min(np.where(r[0] < np.cumsum(a / asum))[0])
    
        tau = np.log(1 / r[1]) / asum
        
        X = X + stochiometry_matrix[:, j]                  # update the no of molecules state
        t = t + tau                                        # update time
        
        
        x1.append(X[0])
        x2.append(X[1])
        x3.append(X[2])
        time.append(t)
        
        if np.round(t)%1000==0:                            # keep track of time
            print(t)

    
    end = tt.time()
    
    print('time = ', end - start, 's')

   
    
    # Save the results as .npy 
    
    df = pd.DataFrame({'time': time, 'NFkB': x1, 'I mRNA': x2, 'I': x3})
    

    np.save('GA_tfinal_%d_iniNn_%d_mol_%d_N_tot_%d_K_Nin_%0.5f_alpha_%0.5f.npy' % (tfinal, int(mol_num), mol_num, N_tot, k_Nin, alpha), df)

	
                                                                                               
    return time, x1, x2, x3



mol_num = 500       # total N_n no of molecules
tfinal = 5000       # time till the Gillespie is to be simulated

# parameters
k_Nin = 5.196
alpha = 1.044

# Call the Gillespie algorithm function
time, x1, x2, x3 = GA(mol_num, tfinal, k_Nin, alpha)

time, x1, x2, x3 = np.array(time), np.array(x1), np.array(x2), np.array(x3)

# plot the results
plt.figure(figsize = (15, 12))
plt.plot(time, x1, lw = 2, label = r'nuclear $ NF \kappa B$ $N_n$')
plt.plot(time , x3/10, label = r'cytoplasmic $I\kappa B$ $I$')
plt.legend(fontsize = 20)
plt.xlabel('time (min)', fontsize  = 20)
plt.ylabel(r'No of molecules ($NF\kappa B$, $\times$ 10 $I\kappa B$)', fontsize = 20)
plt.xticks(np.arange(0, 5100, 250), fontsize = 12, rotation = 45)
plt.yticks(np.arange(0, 2200, 100), fontsize = 12)
plt.title('3 Variable Model', fontsize = 25, pad = 15)

plt.show()







