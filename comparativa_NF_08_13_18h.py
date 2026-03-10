#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% Obtengo ciclos y resultados para cada concentracion - Todo a 300 kHz
# 08 hs
ciclos_08 = glob("NF08hs/**/**/**/*ciclo_promedio_H_M.txt") 
resultados_08 = glob("NF08hs/**/**/**/*resultados.txt")
ciclos_08.sort()
resultados_08.sort()
conc_08 = 41.0 #g/L

# 13 hs
ciclos_13 = glob("NF13hs/**/**/**/*ciclo_promedio_H_M.txt")
resultados_13 = glob("NF13hs/**/**/**/*resultados.txt")
ciclos_13.sort()
resultados_13.sort()
conc_13 = 26.7 #g/L

# 18 hs
ciclos_18 = glob("NF18hs/**/**/**/*ciclo_promedio_H_M.txt")
resultados_18 = glob("NF18hs/**/**/**/*resultados.txt")
ciclos_18.sort()
resultados_18.sort()
conc_18 = 56.4/2 #g/L

for p in ciclos_08:
    print('  ',p)
for p in ciclos_13:
    print('  ',p)
for p in ciclos_18:
    print('  ',p)

print('\n')
for res in resultados_08:
    print('  ',res)
for res in resultados_13:
    print('  ',res)
for res in resultados_18:
    print('  ',res)
    
#%% Ploteo Ciclos Promedio 
fig00, (ax,ax2,ax3) =plt.subplots(1,3,figsize=(15,5),constrained_layout=True,sharey=True,sharex=True)

for i,e in enumerate(ciclos_08):
    if '100dA' in e:
        _,_,_, H_08,M_08,_ = lector_ciclos(ciclos_08[i])
        ax.plot(H_08/1000,M_08,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_08):
    if '125dA' in e:
        _,_,_, H_08,M_08,_ = lector_ciclos(ciclos_08[i])
        ax2.plot(H_08/1000,M_08,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_08):
    if '150dA' in e:
        _,_,_, H_08,M_08,_ = lector_ciclos(ciclos_08[i])
        ax3.plot(H_08/1000,M_08,'-',label=f'NF{i}')

ax.set_ylabel('M (A/m)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')
ax3.set_title('57 kA/m',loc='left')

for a in ax,ax2,ax3:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left')
plt.suptitle('Comparativa ciclos promedio NF 08 hs\n300 kHz')

fig01, (ax,ax2,ax3) =plt.subplots(1,3,figsize=(15,5),constrained_layout=True,sharey=True,sharex=True)

for i,e in enumerate(ciclos_13):
    if '100dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13[i])
        ax.plot(H_13/1000,M_13,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_13):
    if '125dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13[i])
        ax2.plot(H_13/1000,M_13,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_13):
    if '150dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13[i])
        ax3.plot(H_13/1000,M_13,'-',label=f'NF{i}')

ax.set_ylabel('M (A/m)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')
ax3.set_title('57 kA/m',loc='left')

for a in ax,ax2,ax3:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left')
plt.suptitle('Comparativa ciclos promedio NF 13 hs\n300 kHz')

fig020, (ax,ax2,ax3) =plt.subplots(1,3,figsize=(15,5),constrained_layout=True,sharey=True,sharex=True)

for i,e in enumerate(ciclos_18):
    if '100dA' in e:
        _,_,_, H_18,M_18,_ = lector_ciclos(ciclos_18[i])
        ax.plot(H_18/1000,M_18,'-',label=f'NF{str(i).zfill(2)}')

for i,e in enumerate(ciclos_18):
    if '110dA' in e:
        _,_,_, H_18,M_18,_ = lector_ciclos(ciclos_18[i])
        ax2.plot(H_18/1000,M_18,'-',label=f'NF{str(i).zfill(2)}')

for i,e in enumerate(ciclos_18):
    if '120dA' in e:
        _,_,_, H_18,M_18,_ = lector_ciclos(ciclos_18[i])
        ax3.plot(H_18/1000,M_18,'-',label=f'NF{str(i).zfill(2)}')

ax.set_ylabel('M (A/m)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('42 kA/m',loc='left')
ax3.set_title('45 kA/m',loc='left')

for a in ax,ax2,ax3:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left')
plt.suptitle('Comparativa ciclos promedio NF 18 hs\n300 kHz')


fig021, (ax,ax2,ax3) =plt.subplots(1,3,figsize=(15,5),constrained_layout=True,sharey=True,sharex=True)

for i,e in enumerate(ciclos_18):
    if '130dA' in e:
        _,_,_, H_18,M_18,_ = lector_ciclos(ciclos_18[i])
        ax.plot(H_18/1000,M_18,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_18):
    if '140dA' in e:
        _,_,_, H_18,M_18,_ = lector_ciclos(ciclos_18[i])
        ax2.plot(H_18/1000,M_18,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_18):
    if ('150dA' in e) and not ('full' in e):
        _,_,_, H_18,M_18,_ = lector_ciclos(ciclos_18[i])
        ax3.plot(H_18/1000,M_18,'-',label=f'NF{i}')

ax.set_ylabel('M (A/m)')
ax.set_title('50 kA/m',loc='left')
ax2.set_title('53 kA/m',loc='left')
ax3.set_title('57 kA/m',loc='left')

for a in ax,ax2,ax3:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left')
plt.suptitle('Comparativa ciclos promedio NF 18 hs\n300 kHz')

#%% Salvo comparativa ciclos promedio
figs=[fig00,fig01,fig020,fig021]
names=['ciclos_promedio_NF08h_38_47_57',
       'ciclos_promedio_NF13h_38_47_57',
       'ciclos_promedio_NF18h_38_42_45',
       'ciclos_promedio_NF18h_50_53_57']

for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'0_{e[1]}.png',dpi=300)

#%% Listas con Resultados
res_08=[]
print('Resultados 08 hs', '='*80,'\n')
for r in resultados_08:
    res_08.append(ResultadosESAR(os.path.dirname(r)))

res_13=[]
print('Resultados 13 hs', '='*80,'\n')
for r in resultados_13:
    res_13.append(ResultadosESAR(os.path.dirname(r)))
    
res_18=[]
print('Resultados 18 hs', '='*80,'\n')
for r in resultados_18:
    if not 'full' in r:
        res_18.append(ResultadosESAR(os.path.dirname(r)))
    

#%% 1- Templogs
# 8 hs
fig10, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,9),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_08):
    if '100' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_08):
    if '125' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax2.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_08):
    if '150' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax3.plot(r.time,r.temperatura,'.-',label=label)

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper left')
    a.set_ylabel('T (ºC)')
ax.set_xlim(0,40)
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')    

ax3.set_xlabel('t (s)')
plt.suptitle(f'Templogs\nNF 08 hs - {conc_08:.1f} g/L')

#% 13 hs
fig11, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,9),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_13):
    if '100' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_13):
    if '125' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax2.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_13):
    if '150' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax3.plot(r.time,r.temperatura,'.-',label=label)

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='lower right')
    a.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')    
ax3.set_xlabel('t (s)')
plt.suptitle(f'Templogs\nNF 13 hs - {conc_13:.1f} g/L')

#% 18 hs
fig100, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,9),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_18):
    if '100dA' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_18):
    if '110dA' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax2.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_18):
    if '120dA' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax3.plot(r.time,r.temperatura,'.-',label=label)

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='lower right')
    a.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.set_title('38 kA/m',loc='left')
ax2.set_title('42 kA/m',loc='left')    
ax3.set_title('45 kA/m',loc='left')    
ax3.set_xlabel('t (s)')
plt.suptitle(f'Templogs\nNF 18 hs - {conc_18:.1f} g/L')

fig101, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,9),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_18):
    if '130dA' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{str(i).zfill(2)}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_18):
    if '140dA' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{i}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax2.plot(r.time,r.temperatura,'.-',label=label)

for i,r in enumerate(res_18):
    if '150dA' in r.directorio:
        dt = r.time[-1]-r.time[0]
        dT = r.temperatura[-1]-r.temperatura[0]
        rate=dT/dt
        label=f'NF{i}  $\Delta$t= {dt:.2f} s  $\Delta$T= {dT:.2f} °C  WR= {rate:.2f} °C/s'
        ax3.plot(r.time,r.temperatura,'.-',label=label)

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='lower right')
    a.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.set_title('50 kA/m',loc='left')
ax2.set_title('53 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')    
ax3.set_xlabel('t (s)')
plt.suptitle(f'Templogs\nNF 18 hs - {conc_18:.1f} g/L')

#%% Salvo templogs
figs=[fig10,fig11,fig100,fig101]
names=['Templog_NF08h_38_47_57',
       'Templog_NF13h_38_47_57',
       'Templog_NF18h_38_42_45',
       'Templog_NF18h_50_53_57']

for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'1_{e[1]}.png',dpi=300)

#%% 2 - Tau vs time / Temp
# 8 hs
fig200, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_08):
    if '100dA' in r.directorio:
        ax.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '125dA' in r.directorio:
        ax2.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '150dA' in r.directorio:
        ax3.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('τ (ns)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')   
ax3.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
plt.suptitle(f'tau vs time\nNF 8 hs - {conc_08:0.1f} g/L' )

fig201, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_08):
    if '100dA' in r.directorio:
        ax.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '125dA' in r.directorio:
        ax2.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '150dA' in r.directorio:
        ax3.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('τ (ns)')
ax3.set_xlabel('T (°C)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  

plt.suptitle(f'tau vs Temperatura\nNF 8 hs - {conc_08:0.1f} g/L' )

#% 13 hs

fig210, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '100dA' in r.directorio:
        ax.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '125dA' in r.directorio:
        ax2.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '150dA' in r.directorio:
        ax3.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('τ (ns)')
ax.set_xlim(0,)

ax3.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
plt.suptitle(f'tau vs time\nNF 13 hs - {conc_13:0.1f} g/L' )

fig211, (ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '100dA' in r.directorio:
        ax.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '125dA' in r.directorio:
        ax2.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '150dA' in r.directorio:
        ax3.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('τ (ns)')
ax3.set_xlabel('T (°C)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  

plt.suptitle(f'tau vs Temperatura\nNF 13 hs - {conc_13:0.1f} g/L' )
 
#% 18 hs 
fig220, (ax,ax2,ax3,ax4,ax5,ax6) =plt.subplots(6,1,figsize=(10,10),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_18):
    if '100dA' in r.directorio:
        ax.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '110dA' in r.directorio:
        ax2.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '120dA' in r.directorio:
        ax3.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '130dA' in r.directorio:
        ax4.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '140dA' in r.directorio:
        ax5.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')
        
for i,r in enumerate(res_18):
    if '150dA' in r.directorio:
        ax6.plot(r.time,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3,ax4,ax5,ax6:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('τ (ns)')
ax.set_xlim(0,)

ax6.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('42 kA/m',loc='left')    
ax3.set_title('45 kA/m',loc='left')  
ax4.set_title('50 kA/m',loc='left')
ax5.set_title('53 kA/m',loc='left')    
ax6.set_title('57 kA/m',loc='left')  
plt.suptitle(f'tau vs time\nNF 18 hs - {conc_18:0.1f} g/L' )

fig221, (ax,ax2,ax3,ax4,ax5,ax6) =plt.subplots(6,1,figsize=(10,10),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_18):
    if '100dA' in r.directorio:
        ax.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '110dA' in r.directorio:
        ax2.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '120dA' in r.directorio:
        ax3.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '130dA' in r.directorio:
        ax4.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '140dA' in r.directorio:
        ax5.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')
        
for i,r in enumerate(res_18):
    if '150dA' in r.directorio:
        ax6.plot(r.temperatura,r.tau,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3,ax4,ax5,ax6:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('τ (ns)')
ax6.set_xlabel('T (°C)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('42 kA/m',loc='left')    
ax3.set_title('45 kA/m',loc='left')  
ax4.set_title('50 kA/m',loc='left')
ax5.set_title('53 kA/m',loc='left')    
ax6.set_title('57 kA/m',loc='left')  
plt.suptitle(f'tau vs Temperatura\nNF 18 hs - {conc_18:0.1f} g/L' )
#%% Salvo figs tau
figs=[fig200,fig201,fig210,fig211,fig220,fig221]
names=['tau_vs_time_NF08h_38_47_57',
       'tau_vs_Temp_NF08h_38_47_57',
       'tau_vs_time_NF13h_38_47_57',
       'tau_vs_Temp_NF13h_38_47_57',
       'tau_vs_time_NF18h_38_42_45_50_53_57',
       'tau_vs_Temp_NF18h_38_42_45_50_53_57',
    ]   

for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'2_{e[1]}.png',dpi=300)

#%% 3 - ESAR vs time / Temp
# 8 hs
fig300, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_08):
    if '100dA' in r.directorio:
        ax.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '125dA' in r.directorio:
        ax2.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '150dA' in r.directorio:
        ax3.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('SAR (W/g)')
ax.set_xlim(0,)
ax3.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
plt.suptitle(f'ESAR vs time\nNF 8 hs - {conc_08:0.1f} g/L' )

fig301, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_08):
    if '100dA' in r.directorio:
        ax.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '125dA' in r.directorio:
        ax2.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_08):
    if '150dA' in r.directorio:
        ax3.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('SAR (W/g)')
ax3.set_xlabel('T (°C)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  

plt.suptitle(f'ESAR vs Temperatura\nNF 8 hs - {conc_08:0.1f} g/L' )

#% 13 hs

fig310, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '100dA' in r.directorio:
        ax.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '125dA' in r.directorio:
        ax2.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '150dA' in r.directorio:
        ax3.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('SAR (W/g)')
ax.set_xlim(0,)

ax3.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  
plt.suptitle(f'ESAR vs time\nNF 13 hs - {conc_13:0.1f} g/L' )

fig311, (ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_13):
    if '100dA' in r.directorio:
        ax.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '125dA' in r.directorio:
        ax2.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_13):
    if '150dA' in r.directorio:
        ax3.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('SAR (W/g)')
ax3.set_xlabel('T (°C)')
ax.set_title('38 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')    
ax3.set_title('57 kA/m',loc='left')  

plt.suptitle(f'ESAR vs Temperatura\nNF 13 hs - {conc_13:0.1f} g/L' )
 
#% 18 hs 
fig320, (ax,ax2,ax3,ax4,ax5,ax6) =plt.subplots(6,1,figsize=(10,10),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_18):
    if '100dA' in r.directorio:
        ax.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '110dA' in r.directorio:
        ax2.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '120dA' in r.directorio:
        ax3.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '130dA' in r.directorio:
        ax4.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '140dA' in r.directorio:
        ax5.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')
        
for i,r in enumerate(res_18):
    if '150dA' in r.directorio:
        ax6.plot(r.time,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3,ax4,ax5,ax6:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('SAR (W/g)')
ax.set_xlim(0,)

ax6.set_xlabel('t (s)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('42 kA/m',loc='left')    
ax3.set_title('45 kA/m',loc='left')  
ax4.set_title('50 kA/m',loc='left')
ax5.set_title('53 kA/m',loc='left')    
ax6.set_title('57 kA/m',loc='left')  
plt.suptitle(f'ESAR vs time\nNF 18 hs - {conc_18:0.1f} g/L' )

fig321, (ax,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(10,10),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_18):
    if '100dA' in r.directorio:
        ax.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '110dA' in r.directorio:
        ax2.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '120dA' in r.directorio:
        ax3.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '130dA' in r.directorio:
        ax4.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for i,r in enumerate(res_18):
    if '140dA' in r.directorio:
        ax5.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')
        
for i,r in enumerate(res_18):
    if '150dA' in r.directorio:
        ax6.plot(r.temperatura,r.SAR,'.-',label=f'NF{str(i).zfill(2)}')

for a in ax,ax2,ax3,ax4,ax5,ax6:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('SAR (W/g)')
ax6.set_xlabel('T (°C)')

ax.set_title('38 kA/m',loc='left')
ax2.set_title('42 kA/m',loc='left')    
ax3.set_title('45 kA/m',loc='left')  
ax4.set_title('50 kA/m',loc='left')
ax5.set_title('53 kA/m',loc='left')    
ax6.set_title('57 kA/m',loc='left')  
plt.suptitle(f'ESAR vs Temperatura\nNF 18 hs - {conc_18:0.1f} g/L' )

#%% Salvo graficos SAR

figs=[fig300,fig301,fig310,fig311,fig320,fig321]
names=['SAR_vs_time_NF08h_38_47_57',
       'SAR_vs_Temp_NF08h_38_47_57',
       'SAR_vs_time_NF13h_38_47_57',
       'SAR_vs_Temp_NF13h_38_47_57',
       'SAR_vs_time_NF18h_38_42_45_50_53_57',
       'SAR_vs_Temp_NF18h_38_42_45_50_53_57',
    ]   

for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'3_{e[1]}.png',dpi=300)




#%% 4 - Hc vs time/Temp

fig40, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.time,r.Hc,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.time,r.Hc,'.-',label=f'NF {i}')

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('Hc (A/m)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('Hc NF vs time - 5 g/L & 10 g/L')

fig41, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=True,sharex=True)

for i,r in enumerate(res_05):
    ax.plot(r.temperatura,r.Hc,'.-',label=f'NF {i}')

for i,r in enumerate(res_10):
    ax2.plot(r.temperatura,r.Hc,'.-',label=f'NF {i}')

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('Hc (A/m)')
ax.set_xlim(24,75)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('T (°C)')
plt.suptitle('Hc NF vs temperatura - 5 g/L & 10 g/L')
#%% 5 - Mag Max vs time/Temp

fig50, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_05):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i}  $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax.plot(r.time,r.mag_max,'.-',label=label)

for i,r in enumerate(res_10):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax2.plot(r.time,r.mag_max,'.-',label=label)

for a in ax,ax2:
    a.grid()
    a.legend(loc='best')
    a.set_ylabel('M$_{max}$ (A/m)')
ax.set_xlim(0,)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('t (s)')
plt.suptitle('M$_{max}$ NF vs time - 5 g/L & 10 g/L')


fig51, (ax,ax2) =plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharey=False,sharex=True)

for i,r in enumerate(res_05):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax.plot(r.temperatura,r.mag_max,'.-',label=label)

for i,r in enumerate(res_10):
    dM = r.mag_max[-1]-r.mag_max[0]
    label=f'NF{i} - $\Delta$'+'M$_{max}$ = '+f'{dM:.1f} A/m '
    ax2.plot(r.temperatura,r.mag_max,'.-',label=label)

for a in ax,ax2:
    a.grid()
    a.legend(loc='upper right')
    a.set_ylabel('M$_{max}$ (A/m)')
ax.set_xlim(24,75)
ax.set_title('NF - 5 g/L',loc='left')
ax2.set_title('NF - 10 g/L',loc='left')    
ax2.set_xlabel('T (°C)')
plt.suptitle('M$_{max}$ NF vs temperatura - 5 g/L & 10 g/L')


# %% Salvo las figuras 
figs=[fig0,fig1,fig20,fig21,fig30,fig31,fig40,fig41,fig50,fig51]
names=['ciclos_promedio_NF5_NF10',
       'templog_NF5_NF10',
       'tau_vs_time_NF5_NF10',
       'tau_vs_Temp_NF5_NF10',
       'SAR_vs_time_NF5_NF10',
       'SAR_vs_Temp_NF5_NF10',
       'Hc_vs_time_NF5_NF10',
       'Hc_vs_Temp_NF5_NF10',
       'Mmax_vs_time_NF5_NF10',
       'Mmax_vs_Temp_NF5_NF10' ]


for i,e in enumerate(zip(figs,names)):
    e[0].savefig(f'{i}_{e[1]}.png',dpi=300)

# %%
