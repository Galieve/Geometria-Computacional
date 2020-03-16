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


def open_file(filename):
    workpath = os.path.dirname(os.path.abspath(__file__))
    # os.getcwd()
    f = nc.netcdf_file(workpath + "/" + filename, 'r')
    return f


# el offset sirve para algo, CUIDADO
def get_values(f, field):
    return f.variables[field][:].copy()


# def generate_variables_hgt(f):
#     time = f.variables['time'][:].copy()
#     time_bnds = f.variables['time_bnds'][:].copy()
#     time_units = f.variables['time'].units
#     level = f.variables['level'][:].copy()
#     lats = f.variables['lat'][:].copy()
#     lons = f.variables['lon'][:].copy()
#     hgt = f.variables['hgt'][:].copy()
#     hgt_units = f.variables['hgt'].units
#     hgt_scale = f.variables['hgt'].scale_factor
#     hgt_offset = f.variables['hgt'].add_offset
#     f.close()


def get_idx(list, value):
    idx = 0
    for i in list:
        if i == value:
            break
        idx += 1
    return idx


def exercise1(f):
    time = get_values(f, 'time')
    lats = get_values(f, 'lat')
    lons = get_values(f, 'lon')
    hgt = get_values(f, 'hgt')
    level = get_values(f, 'level')
    f.close()

    p_idx = get_idx(level, 500)
    X = hgt[:, p_idx, :, :].reshape(len(time), len(lats) * len(lons))
    # Y = X.transpose()

    n_components = 4
    pca = PCA(n_components=n_components)
    pca.fit(X)
    print("The explained variance ratio is:", pca.explained_variance_ratio_)
    print("The total explained variance ratio is:", sum(pca.explained_variance_ratio_))

    element_pca = pca.components_
    element_pca = element_pca.reshape(n_components, len(lats), len(lons))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 5):
        ax = fig.add_subplot(2, 2, i)
        ax.text(0.5, 90, 'PCA-' + str(i),
                fontsize=18, ha='center')
        plt.contour(lons, lats, element_pca[i - 1, :, :])
    plt.show()


def analogous(a, b, f_idx, s_idx):
    a = a.astype('int64')
    b = b.astype('int64')
    res = 0
    res += 0.5 * np.sum((a[f_idx, :, :] - b[f_idx, :, :]) ** 2)
    res += 0.5 * np.sum((a[s_idx, :, :] - b[s_idx, :, :]) ** 2)
    return np.sqrt(res)


def get_analogous_days(a0):
    f = open_file("hgt.2019.nc")
    time = get_values(f, 'time')
    lats = get_values(f, 'lat')
    lons = get_values(f, 'lon')
    hgt = get_values(f, 'hgt')
    level = get_values(f, 'level')
    lons = np.array([i if i <= 180 else i - 360 for i in lons])
    f.close()

    f = open_file("hgt.2020.nc")
    time20 = get_values(f, 'time')
    time_idx = get_idx(time20, a0)
    hgta_0 = get_values(f, 'hgt')[time_idx]
    f.close()

    f_idx = get_idx(level, 500)
    s_idx = get_idx(level, 1000)

    hgta_0 = (hgta_0[:, (30 < lats) & (lats < 50), :])[:, :, (-20 < lons) & (lons < 20)]
    hgt_copy = hgt.copy()
    hgt_copy = (hgt_copy[:, :, (30 < lats) & (lats < 50), :])[:, :, :, (-20 < lons) & (lons < 20)]
    analogous_time = [analogous(hgta_0, hgt_copy[i], f_idx, s_idx) for i in
                      range(len(time))]

    anag_time_arg_sort = np.array(analogous_time).argsort()
    best_idx = [anag_time_arg_sort[i] for i in range(4)]
    # print(best_idx, [analogous_time[i] for i in best_idx])
    time_sim = [time[i] for i in best_idx]
    print("The best days are:", [(dt.date(1800, 1, 1) + dt.timedelta(hours=t)).strftime("%d/%m/%Y") for t in time_sim])
    return best_idx


def exercise2():
    a0 = (dt.date(2020, 1, 20) - dt.date(1800, 1, 1)).days * 24
    analogous_days = get_analogous_days(a0)

    f = open_file("air.2020.nc")
    time20 = get_values(f, 'time')
    time_a0_idx = get_idx(time20, a0)
    temp_a0 = get_values(f, 'air')[time_a0_idx]
    f.close()

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


    p_idx = get_idx(level, 1000)

    temp = temp[:, p_idx, :, :] * scale_factor + offset
    temp_a0 = temp_a0[p_idx, :, :] * scale_factor + offset

    temp = np.array((temp[:, (30 < lats) & (lats < 50), :])[:, :, (-20 < lons) & (lons < 20)])
    temp_a0 = np.array((temp_a0[(30 < lats) & (lats < 50), :])[:, (-20 < lons) & (lons < 20)])

    avg_temp = sum(temp[:, :, :]) / len(temp)

    temp_error = abs(temp_a0 - avg_temp)

    avg_error = np.sum(temp_error) / temp_error.size

    print("The average error is:", avg_error, units)


if __name__ == "__main__":
    f = open_file("hgt.2019.nc")
    exercise1(f)
    exercise2()
