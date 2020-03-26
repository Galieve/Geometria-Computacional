# -*- coding: utf-8 -*-
"""
Referencias:

    Fuente primaria del reanálisis
    https://www.esrl.noaa.gov/psd/data/gridded/
        data.ncep.reanalysis2.pressure.html

    Altura geopotencial en niveles de presión
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/
        DBListFiles.pl?did=59&tid=81620&vid=1498

    Temperatura en niveles de presión:
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/
        DBListFiles.pl?did=59&tid=81620&vid=1497

"""

import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

# Establecemos como directorio la carpeta de este archivo.
def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)

# Devolvemos un archivo en la ruta relativa adecuada.
def open_file(filename):
    # os.getcwd()
    f = nc.netcdf_file(filename, 'r')
    return f


# Dado un campo y un archivo, devolvemos el campo del archivo pedido.
def get_values(f, field):
    return f.variables[field][:].copy()


# Dada una lista (no necesariamente ordenada) y un valor, buscamos
# el indice de la lista en la que se encuentra este valor.
def get_idx(list, value):
    idx = 0
    for i in list:
        if i == value:
            break
        idx += 1
    return idx


def exercise1():
    # Abrimos el fichero hgt.2019.nc y obtenemos los atributos que
    # vamos a usar
    f = open_file("hgt.2019.nc")
    time = get_values(f, 'time')
    lats = get_values(f, 'lat')
    lons = get_values(f, 'lon')
    hgt = get_values(f, 'hgt')
    level = get_values(f, 'level')
    f.close()

    # X es el sistema de dias -> alturas geopotenciales del aire a 500 hPa.
    p_idx = get_idx(level, 500)
    X = hgt[:, p_idx, :, :].reshape(len(time), len(lats) * len(lons))

    # Realizamos el análisis PCA
    n_components = 4
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("The explained variance ratio is:",
          [f'{vr:.6f}' for vr in pca.explained_variance_ratio_])
    print("The total explained variance ratio is:",
          f'{sum(pca.explained_variance_ratio_):.6f}')

    element_pca = pca.components_
    element_pca = element_pca.reshape(n_components, len(lats), len(lons))

    # Para cada una de las cuatro componentes, pintamos las curvas de nivel.
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        ax.text(0.5, 90, 'PCA-' + str(i),
                fontsize=18, ha='center')
        map = plt.contour(lons, lats, element_pca[i - 1, :, :],
                          cmap=plt.get_cmap('hsv'))
        plt.colorbar(map)



    plt.savefig("PCAs.png")
    # plt.show()


# Dados a, b y dos indices f_idx y s_idx, obtenemos el valor de analogia
# de a y b con la distancia Euclidea ponderada
def analogous(a, b, f_idx, s_idx):
    a = a.astype('int64')
    b = b.astype('int64')
    res = 0
    res += 0.5 * np.sum((a[f_idx, :, :] - b[f_idx, :, :]) ** 2)
    res += 0.5 * np.sum((a[s_idx, :, :] - b[s_idx, :, :]) ** 2)
    return np.sqrt(res)


# Dado a0, obtenemos los indices de los dias análogos.
def get_analogous_days(a0):

    # Abrimos el archivo hgt.2019.nc y obtenemos los datos que vamos a usar.
    f = open_file("hgt.2019.nc")
    time = get_values(f, 'time')
    lats = get_values(f, 'lat')
    lons = get_values(f, 'lon')
    hgt = get_values(f, 'hgt')
    level = get_values(f, 'level')
    f.close()

    # Realizamos el cambio de coordenadas propuesto.
    lons = np.array([i if i <= 180 else i - 360 for i in lons])

    # Abrimos el archivo hgt.2020.nc y obtenemos los datos que vamos a usar.
    f = open_file("hgt.2020.nc")
    time20 = get_values(f, 'time')
    time_idx = get_idx(time20, a0)
    hgta_0 = get_values(f, 'hgt')[time_idx]
    f.close()

    # Buscamos los indices en el array asociados a 500 hPa y 1000 hPa.
    f_idx = get_idx(level, 500)
    s_idx = get_idx(level, 1000)

    # Obtenemos los datos de altura de nuestro subconjunto S
    hgta_0 = hgta_0[:, (30 < lats) & (lats < 50), :]
    hgta_0 = hgta_0[:, :, (-20 < lons) & (lons < 20)]
    hgt_copy = hgt.copy()
    hgt_copy = hgt_copy[:, :, (30 < lats) & (lats < 50), :]
    hgt_copy = hgt_copy[:, :, :, (-20 < lons) & (lons < 20)]

    # Obtenemos la lista de valores asociados a cada dia y a0 mediante
    # la función analogous
    analogous_time = [analogous(hgta_0, hgt_copy[i], f_idx, s_idx) for i in
                      range(len(time))]

    # Ordenamos los indices del array según los valores de analogous_time
    # y calculamos a que dia se corresponden.
    anag_time_arg_sort = np.array(analogous_time).argsort()
    best_idx = anag_time_arg_sort[0:4]
    time_sim = [time[i] for i in best_idx]
    print("The best days are:",
          [(dt.date(1800, 1, 1) +
            dt.timedelta(hours=t)).strftime("%d/%m/%Y") for t in time_sim])
    
    # Devolvemos los mejores dias para utilizarlos luego
    return best_idx


def exercise2():
    
    # Obtenemos los dias analogos a a0
    a0 = (dt.date(2020, 1, 20) - dt.date(1800, 1, 1)).days * 24
    analogous_days = get_analogous_days(a0)

    # Abrimos el archivo air.2020.nc y obtenemos los datos que vamos a usar.
    f = open_file("air.2020.nc")
    time20 = get_values(f, 'time')
    time_a0_idx = get_idx(time20, a0)
    temp_a0 = get_values(f, 'air')[time_a0_idx]
    f.close()

    # Abrimos el archivo air.2019.nc y obtenemos los datos que vamos a usar.
    f = open_file("air.2019.nc")
    lats = get_values(f, 'lat')
    lons = get_values(f, 'lon')
    lons = np.array([i if i <= 180 else i - 360 for i in lons])
    level = get_values(f, 'level')
    temp = get_values(f, 'air')[analogous_days]
    scale_factor = f.variables['air'].scale_factor
    offset = f.variables['air'].add_offset
    units = f.variables['air'].units
    f.close()

    # Calculamos las temperaturas a 1000 hPa de a0 y de sus analogos
    p_idx = get_idx(level, 1000)
    temp = temp[:, p_idx, :, :] * scale_factor + offset
    temp_a0 = temp_a0[p_idx, :, :] * scale_factor + offset

    # Obtenemos las temperaturas de S
    temp = np.array((temp[:, (30 < lats) & (lats < 50), :])
                    [:, :, (-20 < lons) & (lons < 20)])
    temp_a0 = np.array((temp_a0[(30 < lats) & (lats < 50), :])
                       [:, (-20 < lons) & (lons < 20)])

    # Hallamos el error medio del dia a0 frente a la media de sus analogos.
    avg_temp = sum(temp[:, :, :]) / len(temp)
    temp_error = abs(temp_a0 - avg_temp)
    avg_error = np.sum(temp_error) / temp_error.size

    print("The average error is:", f'{avg_error:.6f}', units)


if __name__ == "__main__":
    setup()
    exercise1()
    exercise2()
