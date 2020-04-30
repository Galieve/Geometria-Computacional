# Nos colocamos en la carpeta de este archivo.
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color
from scipy.spatial import ConvexHull


def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


def get_figure():
    # Parametrización discreta de la banda de Mobius.
    t = np.linspace(-0.5, 0.5, 120)
    theta = np.linspace(0, 2 * np.pi, 120)

    x = np.outer(np.ones_like(t), np.cos(theta)) - np.outer(t, np.sin(theta/2)*np.cos(theta))
    y = np.outer(np.ones_like(t), np.sin(theta)) - np.outer(t, np.sin(theta/2)*np.sin(theta))
    z = np.outer(t, np.cos(theta/2))
    return z, x, y


def get_centroid(X, Y, Z):
    cx, cy, cz = np.sum(X), np.sum(Y), np.sum(Z)
    return cx / X.size, cy / Y.size, cz / Z.size


def get_diameter(X, Y, Z):
    df = pd.DataFrame({'x': X.flatten(), 'y': Y.flatten(), 'z': Z.flatten()})
    xyz = df.values
    hull = ConvexHull(xyz)

    if len(hull.vertices) < 1e3:
        v = hull.vertices
        d = 0
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                d = max(d, np.linalg.norm(xyz[v[i]]-xyz[v[j]]))
        return d
    else:
        print("convex hull failed.")
        return 10


# Devolvemos f(x, y, z, t), donde (x, y, z) estan en
# S2 \ {(0,0,-1)} y t en [0, 1].
def transform(X, Y, Z, c, d, t):
    v = (d*t, d*t, 0)
    theta = 3 * np.pi*t
    opX, opY, opZ = X - c[0], Y - c[1], Z - c[2]
    opX_ = np.cos(theta) * opX - np.sin(theta) * opY
    opY_ = np.sin(theta) * opX + np.cos(theta) * opY
    opZ_ = opZ
    X, Y, Z = c[0] + opX_, c[1] + opY_, c[2] + opZ_
    X, Y, Z = X + v[0], Y + v[1], Z + v[2]
    return X, Y, Z


# Dado un tiempo t, una colección de puntos de la superficie
# (x, y, z) y de la curva (x_, y_, z_) calculamos f_t para
# cada uno de ellos.
def animate(X, Y, Z, c, d, t):
    xt, yt, zt = transform(X, Y, Z, c, d, t)
    ax = plt.axes(projection='3d')

    # Fijamos los ejes para ver fielmente la deformación.
    ax.set_xlim3d(-1, d+1)
    ax.set_ylim3d(-1, d+1)

    cset = ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                    cmap='jet', edgecolor='none')
    ax.clabel(cset, fontsize=9, inline=1)
    return ax,


def exercise1():

    X, Y, Z = get_figure()
    c = get_centroid(X, Y, Z)
    d = get_diameter(X, Y, Z)

    # Generamos la animacion.
    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig,
                                  lambda t: animate(X, Y, Z, c, d, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("möbius band xyz.gif", fps=10)
    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig,
                                  lambda t: animate(Y, Z, X, c, d, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("möbius band yzx.gif", fps=10)


# (x, y, z) y de la curva (x_, y_, z_) calculamos f_t para
# cada uno de ellos.
def animate_leaf(X, Y, Z, c, d, t):
    xt, yt, zt = transform(X, Y, Z, c, d, t)
    ax = plt.axes(projection='3d')

    # Fijamos los ejes para ver fielmente la deformación.
    ax.set_xlim3d(-1, 2*d+1)
    ax.set_ylim3d(-1, 2*d+1)

    ax.scatter(xt, yt, c=zt, s=0.1, animated=True)
    return ax,

def exercise2():
    img = io.imread('arbol.png')
    dimensions = color.guess_spatial_dimensions(img)
    print(dimensions)
    io.show()
    # io.imsave('arbol2.png',img)

    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    fig = plt.figure(figsize=(5, 5))
    p = plt.contourf(img[:, :, 0], cmap=plt.cm.get_cmap('summer'), levels=np.arange(0, 240, 2))
    plt.axis('off')
    # fig.colorbar(p)

    xyz = img.shape
    print(xyz)

    x = np.arange(0, xyz[0], 1)
    y = np.arange(0, xyz[1], 1)
    xx, yy = np.meshgrid(x, y)
    xx = np.asarray(xx).reshape(-1)
    yy = np.asarray(yy).reshape(-1)
    z = img[:, :, 0]
    zz = np.asarray(z).reshape(-1)

    """
    Consideraremos sólo los elementos con zz < 240 

    Por curiosidad, comparamos el resultado con contourf y scatter!
    """
    # Variables de estado coordenadas
    x0 = xx[zz < 240]
    y0 = yy[zz < 240]
    z0 = zz[zz < 240] / 256.
    print(x0.shape)
    print(y0.shape)
    print(z0.shape)
    c = get_centroid(x0, y0, z0)
    d = get_diameter(x0, y0, z0)


    # Generamos la animacion.
    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig,
                                  lambda t: animate_leaf(x0, y0, z0, c, d, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("leaf.gif", fps=10)




if __name__ == "__main__":
    setup()
    exercise1()
    exercise2()