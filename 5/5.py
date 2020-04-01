# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020

@author: Robert
"""

# from mpl_toolkits import mplot3d

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation


# z / d = tan x, x in (-90, 90)

# (x, y, z) = (d sin t, d cos t, z), donde d**2 + z**2 = 1 (d = +sqrt(x**2 + y**2))
# (d sin t, d cos t, z) -> (sin t, cos t, z / d) = (sin t, cos t, h) (cont en S2 \ polos)
# (sin t, cos t, h) -> r = (h + 1), (r sin t, r cos t) si h >= 0.
# (sin t, cos t, h) ->  r = 1/(1 - h), (r sin t, r cos t) si h <= 0.

# (x, y, z) = (d sin t, d cos t, z) -> (sin t, cos t, tan x) -> ((z/d + 1) sin t, (z/d + 1) cos t), z >= 0.
# (x, y, z) = (d sin t, d cos t, z) -> (sin t, cos t, tan x) -> ((d/(d - z) sin t, (d/(d - z) cos t), z <= 0.

# (x, y, z) = (d sin t, d cos t, z) -> (sin t, cos t, tan x) -> ((tan x + 1) sin t, (tan x + 1) cos t), z >= 0.
# (x, y, z) = (d sin t, d cos t, z) -> (sin t, cos t, tan x) -> ((1/(1 - tan x) sin t, (1/(1 - tan x) cos t), z <= 0.


# p: (0, 1) -> S1\{(0,1)} -> R
# t -> (sen t, cos t) -> sen t/(1 - cos t)
# s' = p-1, R -> (0,1)
# s(x) = (s'(x)-0.5)*pi, s: R -> (-pi/2, pi/2)

# tan(x/(1-z)) no esta bien definida.
# f(x,y,z) = (tan(s(x/(1-z))), tan(s(y/(1-z))))

# f(x,y,z) = (tan(x/(1-z)), tan(y/(1-z))
# BIEN DEFINIDA
# x/(1-z) = +-pi/2 no pasa
# x = (1-z)*(pi/2)
# y**2 = 1 - (pi/2)**2*(1-z)**2 - z**2 = g(z)
# ¿g(z) > 0? Si, a veces.


# SOBREYECTIVA
# sean a, b \in R, queremos x, y, z/ x**2+y**2+z**2=1 y f(x,y,z) = (a,b)
# c = arctan(a), d = arctan(b), s(x) = c*(1-z), s(y) = d*(1-z)
# saldrá...
# x**2 + y**2 + z**2 = 1 =>
# c**2*(1-s(z))**2 + d**2*(1-z)**2 + z**2 =>
# g(s(z)) = z**2(1 + c**2 + d**2) - 2*z*(c**2 + d**2) + (c**2 + d**2 - 1) = 0
# discr(g) = 4(c**2 + d**2)**2 - 4*(c**2 + d**2 + 1)*(c**2 + d**2 - 1) =
# 4
# z = (-b +-2)/2*a) = ((c**2+d**2+-1)/(c**2+d**2 + 1) =>
# z = 1, 1 - 2/(c**2+d**2 + 1), pero 1 NO VALE

# INYECTIVA
# sea f(x,y,z) = (a,b), f(x',y',z') = (a,b) => (x,y,z) = (x', y', z')
#  a = tan(x/(1-z)), b = tan(y/(1-z))


def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


def proj(x, z, z0=1, alpha=1):
    z0 = z * 0 + z0
    eps = 1e-16
    x_trans = x / (abs(z0 - z) ** alpha + eps)
    return (x_trans)
    # Nótese que añadimos un épsilon para evitar dividi entre 0!!


def exercise1():
    u = np.linspace(0, np.pi, 25)
    v = np.linspace(0, 2 * np.pi, 50)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    t2 = np.linspace(-1, 0, 500)

    y2 = t2 * np.sin(127 * t2 / 2)
    z2 = t2 * np.cos(127 * t2 / 2)
    x2 = np.sqrt(1 - z2 ** 2 - y2 ** 2)

    t2 = np.linspace(0, 1, 500)
    a2 = t2 * np.sin(127 * t2 / 2)
    y2 = np.concatenate((y2, a2))
    b2 = t2 * np.cos(127 * t2 / 2)
    z2 = np.concatenate((z2, -b2))
    x2 = np.concatenate((x2, -np.sqrt(1 - a2 ** 2 - b2 ** 2)))
    c2 = x2 + y2

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2, y2, z2, '-b', c="gray", zorder=3)
    ax.set_title('2-sphere');
    plt.savefig("2-sphere.png")
    # plt.show()
    plt.close(fig)

    # NUEVO
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(proj(x, z), proj(y, z), z * 0 + 1, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.plot(proj(x2, z2), proj(y2, z2), 1, '-b', c="gray", zorder=3)
    ax.set_title('Stereographic projection');
    # plt.show()
    plt.savefig("Stereographic projection.png");
    plt.close(fig)


def transform(x, y, z, t):
    l = (1 - t) + abs(-1 - z) * t
    return x / l, y / l, -t + z * (1 - t)


def animate(x, y, z, x_, y_, z_, t):
    xt, yt, zt = transform(x, y, z, t)
    x_t, y_t, z_t = transform(x_, y_, z_, t)
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1, 1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                    cmap='jet', edgecolor='none')

    ax.plot(x_t, y_t, z_t, '-b', c="white", zorder=3)
    return ax,


def exercise2():
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    t2 = np.linspace(-1, 0, 500)

    y2 = t2 * np.sin(127 * t2 / 2)
    z2 = t2 * np.cos(127 * t2 / 2)
    x2 = np.sqrt(1 - z2 ** 2 - y2 ** 2)
    t2 = np.linspace(0, 1, 500)
    a2 = t2 * np.sin(127 * t2 / 2)
    y2 = np.concatenate((y2, a2))
    b2 = t2 * np.cos(127 * t2 / 2)
    z2 = np.concatenate((z2, -b2))
    x2 = np.concatenate((x2, -np.sqrt(1 - a2 ** 2 - b2 ** 2)))
    c2 = x2 + y2

    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig, lambda t: animate(x, y, z, x2, y2, z2, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("projected_sphere.gif", fps=10)


if __name__ == "__main__":
    setup()
    # exercise1()
    exercise2()
