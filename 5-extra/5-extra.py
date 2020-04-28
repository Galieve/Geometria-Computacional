import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation

# Nos colocamos en la carpeta de este archivo.
def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


# Devolvemos f(x, y, z, t), donde (x, y, z) estan en
# S2 \ {(0,0,-1)} y t en [0, 1].
def transform_tan(x, y, z, t):
    l = (1 - t) + np.tan(t * np.arctan(1 + z))
    return x / l, y / l, -t + z * (1 - t)


# Dado un tiempo t, una colección de puntos de la superficie
# (x, y, z) y de la curva (x_, y_, z_) calculamos f_t para
# cada uno de ellos.
def animate(x, y, z, x_, y_, z_, t):

    xt, yt, zt = transform_tan(x, y, z, t)
    x_t, y_t, z_t = transform_tan(x_, y_, z_, t)

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


# Ejercicio 3
def exercise3():
    # Parametrización discreta de la esfera.
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 60)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    # Parametrización discreta de la curva gamma.
    tetha = np.linspace(0, np.pi, 500)

    z2 = np.sin(tetha) * np.sin(17.5 * tetha)
    y2 = np.sin(tetha) * np.cos(17.5 * tetha)
    x2 = np.cos(tetha)

    # Generamos la animacion.
    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig, lambda t: animate(x, y, z, x2, y2, z2, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("projected_sphere_tangent_version.gif", fps=10)


if __name__ == "__main__":
    setup()
    exercise3()
