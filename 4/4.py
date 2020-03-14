# -*- coding: utf-8 -*-
"""
Referencias:
    
    Fuente primaria del reanálisis
    https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.pressure.html
    
    Altura geopotencial en niveles de presión
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1498
    
    Temperatura en niveles de presión:
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1497

"""
import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

workpath = os.path.dirname(os.path.abspath(__file__))
os.getcwd()
#os.chdir(workpath)
files = os.listdir(workpath)


#f = nc.netcdf_file(workpath + "/" + files[0], 'r')
f = nc.netcdf_file(workpath + "/air.2019.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
air = f.variables['air'][:].copy()
air_units = f.variables['air'].units
air_scale = f.variables['air'].scale_factor
air_offset = f.variables['air'].add_offset
print(air.shape)

f.close()

"""
Ejemplo de evolución temporal de un elemento de aire
"""
plt.plot(time, air_offset + air[:, 1, 1, 1]*air_scale, c='r')
plt.show()

#time_idx = 237  # some random day in 2012
# Python and the renalaysis are slightly off in time so this fixes that problem
# offset = dt.timedelta(hours=0)
# List of all times in the file as datetime objects
dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) #- offset\
           for t in time]
np.min(dt_time)
np.max(dt_time)

"""
Distribución espacial de la temperatura en el nivel de 1000hPa, para el primer día
"""
p_idx = 0
for i in level:
    if i == 500:
        break
    p_idx += 1
    
plt.contour(lons, lats, air[1,p_idx,:,:])
plt.show()

air2 = air[:,p_idx,:,:].reshape(len(time),len(lats)*len(lons))
#air3 = air2.reshape(len(time),len(lats),len(lons))

n_components=4


X = air2
Y = air2.transpose()
pca = PCA(n_components=n_components)

pca.fit(X)
print(pca.explained_variance_ratio_)
out = pca.singular_values_

pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_


State_pca = pca.fit_transform(X)
#Ejercicio de la práctica
Element_pca = pca.components_
print(Element_pca.shape)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()




# d(a0, a1) = sqrt(sum_{i,j}(w_500*(z_500_a0 - z_500_a1)**2 + w_100))
#w_k = 0 si k != 500, 1000
