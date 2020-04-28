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


# Nos colocamos en la carpeta de este archivo.
def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


# Proyectamos la componente x de la esfera usando como eje
# proyectante el z. z0 = -1 ya que el polo que vamos a
# eliminar va a ser el (0,...0,-1).
def proj(x, z, z0=-1, alpha=1):
    z0 = z * 0 + z0
    eps = 1e-16
    # Nótese que añadimos un épsilon para evitar dividir entre 0.
    x_trans = x / (abs(z0 - z) ** alpha + eps)
    return (x_trans)


# Ejercicio 1
def exercise1():
    # Parametrización discreta de la esfera.
    u = np.linspace(0, np.pi, 25)
    v = np.linspace(0, 2 * np.pi, 50)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    # Parametrización discreta de la curva gamma.
    tetha = np.linspace(0, np.pi, 500)

    y2 = np.sin(tetha) * np.sin(16 * tetha)
    z2 = np.sin(tetha) * np.cos(16 * tetha)
    x2 = np.cos(tetha)

    # Generamos la figura.
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.set_zlabel('z')

    # Pintamos la esfera.
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    cmap='jet', edgecolor='none')
    # Pintamos la curva.
    ax.plot(x2, y2, z2, '-b', c="white", zorder=3)
    ax.set_title('2-sphere');
    plt.savefig("2-sphere.png")
    # plt.show()
    plt.close(fig)

    # Generamos la figura proyectada.
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlim3d(-1, 1)

    # No fijamos el resto de ejes ya que sino no se ve
    # adecuadamente la figura. Para un análisis en este sentido,
    # el ejercicio siguiente

    # Pintamos la esfera proyectada sobre el plano z = -1.
    ax.plot_surface(proj(x, z), proj(y, z), z * 0 -1, rstride=1,
                    cstride=1, cmap='jet', edgecolor='none')

    # Pintamos la curva gamma sobre el plano z = -1.
    ax.plot(proj(x2, z2), proj(y2, z2), -1, '-b', c="white", zorder=3)
    ax.set_title('Stereographic projection');
    # plt.show()
    plt.savefig("Stereographic projection.png");
    plt.close(fig)


# Devolvemos f(x, y, z, t), donde (x, y, z) estan en
# S2 \ {(0,0,-1)} y t en [0, 1].
def transform(x, y, z, t):
    # abs(-1-z) = (1+z) en S2
    l = (1 - t) + (1 + z) * t
    return x / l, y / l, -t + z * (1 - t)


# Dado un tiempo t, una colección de puntos de la superficie
# (x, y, z) y de la curva (x_, y_, z_) calculamos f_t para
# cada uno de ellos.
def animate(x, y, z, x_, y_, z_, t):
    xt, yt, zt = transform(x, y, z, t)
    x_t, y_t, z_t = transform(x_, y_, z_, t)

    ax = plt.axes(projection='3d')
    # Fijamos los ejes para ver fielmente la deformación.
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # Pintamos la superficie deformada en tiempo t.
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                    cmap='jet', edgecolor='none')

    # Pintamos la curva gamma deformada en tiempo t.
    ax.plot(x_t, y_t, z_t, '-b', c="white", zorder=3)
    return ax,


# Ejercicio 2.
def exercise2():
    # Parametrización discreta de la esfera.
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    # Parametrización discreta de la curva gamma.
    tetha = np.linspace(0, np.pi, 500)

    y2 = np.sin(tetha) * np.sin(16 * tetha)
    z2 = np.sin(tetha) * np.cos(16 * tetha)
    x2 = np.cos(tetha)

    # Generamos la animacion.
    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig,
                                  lambda t: animate(x, y, z, x2, y2, z2, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("projected_sphere.gif", fps=10)


if __name__ == "__main__":
    setup()
    exercise1()
    exercise2()
