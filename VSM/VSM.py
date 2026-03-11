#%% VSM NF@citrato 260306 08, 13 y 18 hs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#%% Funciones
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: list de valores Hc (puede haber más de uno si hay múltiples cruces por cero)
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% NE & NE@citrico
data_08 = np.loadtxt(os.path.join('data','NF@cit_08.txt'), skiprows=12)
H_08 = data_08[:, 0]  # Gauss
m_08 = data_08[:, 1]

conc_08 = 41 #mg/mL Magnetita

conc_08_mm = conc_08/1000 #mg np/mg de solvente
masa_08 = (0.1157-0.0637)*conc_08_mm
m_08_norm = m_08/masa_08 #emu/g

data_13 = np.loadtxt(os.path.join('data','NF@cit_13.txt'), skiprows=12)
H_13 = data_13[:, 0]  # Gauss
m_13 = data_13[:, 1]  # emu

conc_13 = 26.7 #mg/mL

conc_13_mm = conc_13/1000
masa_13 = (0.1256-0.0744)*conc_13_mm
m_13_norm = m_13/masa_13

data_18 = np.loadtxt(os.path.join('data','NF@cit_18.txt'), skiprows=12)
H_18 = data_18[:, 0]  # Gauss
m_18 = data_18[:, 1]  # emu

conc_18 = 56.4 #mg/mL

conc_18_mm = conc_18/1000
masa_18 = (0.1130-0.0609)*conc_18_mm
m_18_norm = m_18/masa_18

#%%
fig, a = plt.subplots( figsize=(8, 6), constrained_layout=True)
a.plot(H_08, m_08, '.-', label='08')
a.plot(H_13, m_13, '.-', label='13')
a.plot(H_18, m_18, '.-', label='18')

a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title('NF@citrato 260306 - Solvotermal')
a.set_xlabel('H (G)')
a.set_ylabel('m (emu)')
plt.show()
#%% NF Normalizadas por masa

fig2, b = plt.subplots( figsize=(8, 6), constrained_layout=True)

b.plot(H_08, m_08_norm,'.-', label=f'NF08 (norm con m = {masa_08:.1e} g)')
b.plot(H_13, m_13_norm,'.-', label=f'NF13 (norm con m = {masa_13:.1e} g)')
b.plot(H_18, m_18_norm,'.-', label=f'NF18 (norm con m = {masa_18:.1e} g)')

#b.plot(H_13, m_13_norm,'.-', label=f'13 ({masa_13:.1e} g)')
b.set_ylabel('m (emu/g)')
b.legend()
b.grid()
b.set_title('NF@cit 260306 - Solvotermal - Normalizado por masa')
b.set_xlabel('H (G)')
b.set_ylabel('m (emu/g)')

axins = inset_axes(b,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=b.transAxes,
    borderpad=0) 


# Volver a graficar las curvas en el inset
axins.plot(H_08, m_08_norm,'.-')
axins.plot(H_13, m_13_norm,'.-')
axins.plot(H_18, m_18_norm,'.-')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20,20)   # rango eje X
axins.set_ylim(-4, 4)   # rango eje Y

axins.grid()
mark_inset(b, axins, loc1=1, loc2=3, fc="none", ec="gray",zorder=4)
plt.savefig('NF@cit_08_13_18.png',dpi=300)
plt.show()
#%% Fitting para cada muestra individualmente

resultados_fit = {}
H_fit_arrays = {}
m_fit_arrays = {}

for nombre, H, m ,mass in [('NF08', H_08, m_08, masa_08),('NF13', H_13, m_13, masa_13),('NF18', H_18, m_18, masa_18)]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=True,mass=mass)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Y - linear_contrib
    resultados_fit[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Y,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':fit.derived_parameters()['m_s'],
                            'fit': fit}
    H_fit_arrays[nombre] = fit.X
    m_fit_arrays[nombre] = fit.Y


#%% Ploteo los fits
fig08, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)
ax.plot(H_08, m_08_norm,'o-', c='C0',alpha=0.4,label=f'\nDatos originales\n')

ax.plot(resultados_fit['NF08']['H_fit'], resultados_fit['NF08']['m_fit'],'.-', c='C1',
        label=f'NF08 fit\nC = {conc_08} mg/mL\nm = {masa_08*1000:.3f} mg\nMs = {resultados_fit["NF08"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF08"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NF@cit - 08 hs ')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_08, m_08_norm,'o-', c='C0',alpha=0.4)
axins.plot(resultados_fit['NF08']['H_fit'], resultados_fit['NF08']['m_fit'],'.-',c='C1')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NF08_fit.png',dpi=300)
plt.show()
#%% 13 
fig13, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)
ax.plot(H_13, m_13_norm,'o-', c='C2',alpha=0.4,label=f'\nDatos originales\n')

ax.plot(resultados_fit['NF13']['H_fit'], resultados_fit['NF13']['m_fit'],'.-', c='C3',
        label=f'NF13 fit\nC = {conc_13} mg/mL\nm = {masa_13*1000:.3f} mg\nMs = {resultados_fit["NF13"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF13"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NF@cit - 13 hs ')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_13, m_13_norm,'o-', c='C2',alpha=0.4)
axins.plot(resultados_fit['NF13']['H_fit'], resultados_fit['NF13']['m_fit'],'.-',c='C3')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NF13_fit.png',dpi=300)
plt.show()

#%%
fig18, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)
ax.plot(H_18, m_18_norm,'o-', c='C4',alpha=0.4,label=f'\nDatos originales\n')

ax.plot(resultados_fit['NF18']['H_fit'], resultados_fit['NF18']['m_fit'],'.-', c='C5',
        label=f'NF18 fit\nC = {conc_18} mg/mL\nm = {masa_18*1000:.3f} mg\nMs = {resultados_fit["NF18"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF18"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NF@cit - 18 hs ')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_18, m_18_norm,'o-', c='C4',alpha=0.4)
axins.plot(resultados_fit['NF18']['H_fit'], resultados_fit['NF18']['m_fit'],'.-',c='C5')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NF18_fit.png',dpi=300)
plt.show()



#%%
fig2, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(H_NF, m_NF_conc_norm,'o-', c='C4',alpha=0.4,label=f'Datos originales\n')
ax.plot(resultados_fit['NF']['H_fit'], resultados_fit['NF']['m_fit'],
        '.-', c='C1',label=f'NF fit\nC = {conc_NF} mg/mL\nm = {masa_NF_conc*1000:.3f} mg\nMs = {resultados_fit["NF"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NF@citrato ')

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_NF, m_NF_norm,'o-', c='C4',alpha=0.4)
axins.plot(resultados_fit['NF']['H_fit'], resultados_fit['NF']['m_fit'],'.-',c='C1')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NF_fit.png',dpi=300)
plt.show()
#%% Comparo fits 
fig, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(resultados_fit['NF08']['H_fit'], resultados_fit['NF08']['m_fit_sin_lineal'],
        '.-', label=f'NF08\nC = {conc_08} mg/mL\nm = {masa_08*1000:.3f} mg\nMs = {resultados_fit["NF08"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF08"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$\n')

ax.plot(resultados_fit['NF13']['H_fit'], resultados_fit['NF13']['m_fit_sin_lineal'],
        '.-', label=f'NF13\nC = {conc_13} mg/mL\nm = {masa_13*1000:.3f} mg\nMs = {resultados_fit["NF13"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF13"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$\n')

ax.plot(resultados_fit['NF18']['H_fit'], resultados_fit['NF18']['m_fit_sin_lineal'],
        '.-', label=f'NF18\nC = {conc_18} mg/mL\nm = {masa_18*1000:.3f} mg\nMs = {resultados_fit["NF18"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit["NF18"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$\n')

ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1)
ax.grid()
ax.set_title('NF@cit - 260306')

axins = inset_axes(ax,width="40%",height="40%",
    loc='lower right',bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,borderpad=0) 

axins.plot(resultados_fit['NF08']['H_fit'], resultados_fit['NF08']['m_fit_sin_lineal'],'.-')
axins.plot(resultados_fit['NF13']['H_fit'], resultados_fit['NF13']['m_fit_sin_lineal'],'.-')
axins.plot(resultados_fit['NF18']['H_fit'], resultados_fit['NF18']['m_fit_sin_lineal'],'.-')


axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig('NF_comparativa_fits_08_13_18.png',dpi=300)
plt.show()
#%%