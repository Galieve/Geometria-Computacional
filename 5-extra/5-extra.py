import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation


def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


def eps():
    return 1e-6


# r(z) = 1/(1 - z) si z >= 0
# r(z) = 1 + z si z <= 0
# theta = atan2(x, y)
def transform_curve(x, y, z, t, alpha=0.1):
    theta = np.arctan2(y, x)
    r = np.array([1 - i if i >= 0 else (1 / (1 + i) ** alpha if i > -1 else 0) for i in z])
    return (1 - t) * x + t * (r * np.sin(theta)), (1 - t) * y + t * (r * np.cos(theta)), (1 - t) * z


def transform_surface(x, y, z, t, alpha=0.1):
    theta = np.arctan2(y, x)
    r = np.array([[1 - i if i >= 0 else (1 / (1 + i) ** alpha if i > -1 else 0) for i in j] for j in z])
    return (1 - t) * x + t * (r * np.sin(theta)), (1 - t) * y + t * (r * np.cos(theta)), (1 - t) * z


def animate(x, y, z, x_, y_, z_, t):
    alpha = 0.2
    xt, yt, zt = transform_surface(x, y, z, t, alpha)
    x_t, y_t, z_t = transform_curve(x_, y_, z_, t, alpha)
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1, 1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                    cmap='jet', edgecolor='none')

    ax.plot(x_t, y_t, z_t, '-b', c="white", zorder=3)
    return ax,


def exercise3():
    u = np.linspace(0, np.pi, 30)
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

    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig, lambda t: animate(x, y, z, x2, y2, z2, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("projected_sphere_new_version.gif", fps=10)


if __name__ == "__main__":
    setup()
    exercise3()
