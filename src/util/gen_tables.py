import numpy as np
import scipy.special as sps
from scipy.optimize import minimize



def gen_vonMises_kzeta(zeta0):
    A = np.sqrt((1.0+zeta0)/(1.0-zeta0))

    def avM_Anisotropy(kzeta):                
        return np.abs( (kzeta*sps.i0(kzeta)/sps.i1(kzeta) - 1.0)**0.5 - A )

    return minimize(avM_Anisotropy, A**2, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x[0]


def gen_boxcar_kzeta(zeta0):
    A = np.sqrt((1.0+zeta0)/(1.0-zeta0))

    def boxcar_Anisotropy(kzeta):                
        return np.abs( np.sin(np.pi/(1.0 + kzeta))/(np.pi/(1.0 + kzeta)) - zeta0 )       

    return minimize(boxcar_Anisotropy, A, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x[0]

def gen_dipole_kzeta(zeta0, scatt_alpha):
    A = np.sqrt((1.0+zeta0)/(1.0-zeta0))
    def dipole_Anisotropy(kzeta):                
        return np.abs( sps.hyp2f1((scatt_alpha + 2.0)/2.0, 0.5, 2.0, -kzeta)/sps.hyp2f1((scatt_alpha + 2.0)/2.0, 1.5, 2.0, -kzeta) - A**2 )  

    return minimize(dipole_Anisotropy, A, method='nelder-mead', options={'xtol': 1e-8, 'disp': False}).x[0]


print("von Mises Anisotropy: ", gen_vonMises_kzeta(3.0/5))
print("box car Anisotropy: ", gen_boxcar_kzeta(3.0/5))
print("dipole Anisotropy: ", gen_dipole_kzeta(3.0/5,5.0/3))


N = 200
zeta0 = np.linspace(0,0.95,N)
alpha = np.linspace(0,2,N)

vonMises = np.zeros(N)
boxcar = np.zeros(N)

A,Z = np.meshgrid(alpha,zeta0)
dipole = np.zeros((N,N))

for i in range(N):
    vonMises[i] = gen_vonMises_kzeta(zeta0[i])
    boxcar[i] = gen_boxcar_kzeta(zeta0[i])

for i in range(N):
    for j in range(N):
        dipole[j,i] = gen_dipole_kzeta(Z[j,i], A[j,i])

#Save tables
header = " zeta0   kzeta       "
fmt = "%.5f"
delim = '      '
np.savetxt("vonMises_kzeta_table.dat", np.vstack((zeta0,vonMises)).T,header=header, fmt=fmt, delimiter=delim)


header = " zeta0   kzeta       "
fmt = "%.5f"
delim = '      '
np.savetxt("boxcar_kzeta_table.dat", np.vstack((zeta0,boxcar)).T,header=header, fmt=fmt, delimiter=delim)


header = "nx %d \nny %d \nzeta0    alpha   kzeta       "%(N,N)
fmt = "%.5f"
delim = '      '
np.savetxt("dipole_kzeta_table.dat", np.vstack((Z.reshape(-1), A.reshape(-1),dipole.reshape(-1))).T,header=header, fmt=fmt, delimiter=delim)
