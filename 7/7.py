import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color
from scipy.spatial import ConvexHull


# Nos colocamos en la carpeta de este archivo.
def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


def get_figure():
    # Parametrizacion discreta de la banda de Mobius.
    t = np.linspace(-0.5, 0.5, 120)
    theta = np.linspace(0, 2 * np.pi, 120)

    x = np.outer(np.ones_like(t), np.cos(theta)) \
        - np.outer(t, np.sin(theta / 2) * np.cos(theta))
    y = np.outer(np.ones_like(t), np.sin(theta)) \
        - np.outer(t, np.sin(theta / 2) * np.sin(theta))
    z = np.outer(t, np.cos(theta / 2))
    return z, x, y


# Para calcular el centroide, sumamos todos los elementos
# de cada uno de los vectores y dividimos por el numero
# de elementos.
def get_centroid(Xi):
    return [np.sum(X) / X.size for X in Xi]


# Para conseguir el diametro, agrupamos las coordenadas en
# tuplas  y hallamos su envolvente convexa, utilizando
# el metodo ConvexHull de la libreria SciPy. Basta comprobar
# estos puntos 2 a 2 para calcular el diametro
def get_diameter(Xi):
    coord = {str(i): X.flatten() for i, X in enumerate(Xi)}
    df = pd.DataFrame.from_dict(coord)
    points = df.values
    hull = ConvexHull(points)

    if len(hull.vertices) < 1e3:
        v = hull.vertices
        d = 0
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                d = max(d, np.linalg.norm(points[v[i]] - points[v[j]]))
        print("Hemos reducido de " + str(len(points)) + " a " + str(v.size)
              + " puntos, es decir, un "+
              f'{100*(1-v.size/len(points)):.2f}' +"% menos de puntos.")
        return d
    # Obtenemos demasiados puntos para calcular
    # el radio.
    else:
        print("Too many points.")
        return 10


# Rutina que realiza la transformacion pedida, en funcion
# de un tiempo t, de tal forma que cuando t=0 obtenemos los
# puntos de partida y cuando t=1, obtenemos la composicion
# de la rotacion y la traslacion.
def transform(X, Y, Z, c, d, t):
    v = (d * t, d * t, 0)
    theta = 3 * np.pi * t

    # Aplicamos la rotacion
    opX, opY, opZ = X - c[0], Y - c[1], Z - c[2]
    opX_ = np.cos(theta) * opX - np.sin(theta) * opY
    opY_ = np.sin(theta) * opX + np.cos(theta) * opY
    opZ_ = opZ
    X, Y, Z = c[0] + opX_, c[1] + opY_, c[2] + opZ_

    # Aplicamos la traslacion
    X, Y, Z = X + v[0], Y + v[1], Z + v[2]
    return X, Y, Z


# Dado un tiempo t, una coleccion de puntos de la superficie
# (x, y, z) calculamos f_t para cada uno de ellos.
def animate(X, Y, Z, c, d, t):
    xt, yt, zt = transform(X, Y, Z, c, d, t)
    ax = plt.axes(projection='3d')

    # Fijamos los ejes para ver fielmente la deformacion.
    ax.set_xlim3d(-1, d + 1)
    ax.set_ylim3d(-1, d + 1)

    cset = ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                           cmap='jet', edgecolor='none')
    ax.clabel(cset, fontsize=9, inline=1)
    return ax,


def exercise1():
    X, Y, Z = get_figure()
    c = get_centroid([X, Y, Z])
    d = get_diameter([X, Y, Z])
    print("El centroide de la banda de Moebius es el punto (" +
          f'{c[0]:.3f}'+", " +f'{c[1]:.3f}'+ ", "+f'{c[2]:.3f}'
          + ") y el diametro es " + f'{d:.3f}'+".")

    # Generamos la animacion.
    fig = plt.figure(figsize=(6, 6))
    ani = animation.FuncAnimation(fig,
                                  lambda t: animate(X, Y, Z, c, d, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("möbius band xyz.gif", fps=10)
    plt.clf()

    fig = plt.figure(figsize=(6, 6))

    # El centroide en este caso basta reordenar
    # los valores de centroide obtenidos en la figura anterior
    # El diametro se mantiene
    c = [c[1], c[2], c[0]]
    ani = animation.FuncAnimation(fig,
                                  lambda t: animate(Y, Z, X, c, d, t),
                                  np.arange(0, 1, 0.0125), interval=80)

    ani.save("möbius band yzx.gif", fps=10)


# Funcion auxiliar que dibuja la banda
# de moebius dado un instante (figura de la memoria)
def plot_figure(t, name):
    X, Y, Z = get_figure()
    c = get_centroid([X, Y, Z])
    d = get_diameter([X, Y, Z])

    xt, yt, zt = transform(X, Y, Z, c, d, t)
    ax = plt.axes(projection='3d')

    # Fijamos los ejes para ver fielmente la deformacion.
    ax.set_xlim3d(-1, d + 1)
    ax.set_ylim3d(-1, d + 1)

    cset = ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                           cmap='jet', edgecolor='none')
    ax.clabel(cset, fontsize=9, inline=1)
    plt.savefig(name + "t=" + str(t) + ".png")


# Calculamos f_t para (x,y,z) coordenadas de la hoja
# cada uno de ellos. Utilizamos una tercera coordenada
# identicamente nula a la hora de calcular la transformación
def animate_leaf(X, Y, Z, c, d, t):
    xt, yt, _ = transform(X, Y, np.zeros_like(X), c, d, t)
    ax = plt.axes(projection='3d')

    # Fijamos los ejes para ver fielmente la deformacion.
    ax.set_xlim3d(-1, 2 * d + 1)
    ax.set_ylim3d(-1, 2 * d + 1)

    ax.scatter(xt, yt, c=Z, s=0.1, animated=True)
    return ax,


def exercise2():
    img = io.imread('arbol.png')
    io.show()
    plt.axis('off')
    # fig.colorbar(p)

    xyz = img.shape

    # Generamos los puntos de la malla
    # de la figura en dos arrays 1-d
    x = np.arange(0, xyz[0], 1)
    y = np.arange(0, xyz[1], 1)
    xx, yy = np.meshgrid(x, y)
    xx = np.asarray(xx).reshape(-1)
    yy = np.asarray(yy).reshape(-1)

    # Seleccionamos el valor de color rojo
    z = img[:, :, 0]
    zz = np.asarray(z).reshape(-1)

    """
    Consideraremos solo los elementos con zz < 240 
    """
    # Variables de estado coordenadas
    x0 = xx[zz < 240]
    y0 = yy[zz < 240]
    z0 = zz[zz < 240] / 256.

    # Hallamos el centroide y el diametro
    c = get_centroid([x0, y0])
    d = get_diameter([x0, y0])

    print("El centroide de la hoja es el punto (" + f'{c[0]:.3f}'+", "
          +f'{c[1]:.3f}'
          + ") y el diametro es " + f'{d:.3f}'+".")
    # Anyadimos un 0 para trabajar en 3 variables
    c.append(0)

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
