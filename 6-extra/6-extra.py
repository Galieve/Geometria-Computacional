import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from scipy.integrate import simps
from skimage import io, color
from matplotlib import animation
from time import process_time


def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


def eps():
    return 10**-5


# La funcion derivada viene dada
# por el sistema de coordenadas de Lorenz
def derivate(q, a, b, c):
    q1, q2, q3 = q
    r1 = a * (q2 - q1)
    r2 = b * q1 - q2 - q1 * q3
    r3 = q1 * q2 - c * q3
    return [r1, r2, r3]


# Metodo de Euler. Se cumple que qi = qi-1 + h f(qi-1)
def explicit_euler(q0, der, h, n):
    q = np.empty([n + 1, 3])
    dq = np.empty([n + 1, 3])
    q[0] = q0
    dq[0] = der(q[0])
    for i in range(1, n + 1):
        q[i] = q[i - 1] + h * dq[i - 1]
        dq[i] = der(q[i])
    return q, dq


def exercise1_2():
    a = 10
    b = 28
    c = 8 / 3
    f = lambda q: derivate(q, a, b, c)
    h = 10 ** -3
    n = int(32 / h)
    # init = np.linspace()
    q0 = [1, 1, 1]
    q, dq = explicit_euler(q0, f, h, n)
    p = dq / 2
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.winter(np.linspace(0, 1, n+1)))
    colors = plt.cm.jet(np.linspace(0, 1, len(q[:, 0])))
    ax = plt.axes(projection='3d')
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=colors, linewidths=1, s=0.5)
    plt.savefig("q(1,1,1).png")
    ax = plt.axes(projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, linewidths=1, s=0.5)
    plt.savefig("p(1,1,1).png")


def exercise3():
    a = 10
    b = 28
    c = 8 / 3
    f = lambda q: derivate(q, a, b, c)
    h = 10 ** -3
    n = int(32 / h)
    I = np.linspace(-1., 1., num=5, endpoint=True)

    # Generamos un mallado del cubo [-1,1]^3
    q0x, q0y, q0z = np.meshgrid(I, I, I)
    q0x = q0x.flatten()
    q0y = q0y.flatten()
    q0z = q0z.flatten()

    figq = plt.figure()
    axq = figq.add_subplot(1, 1, 1, projection="3d")
    figp = plt.figure()
    axp = figp.add_subplot(1, 1, 1, projection="3d")
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    colors = plt.cm.jet(np.linspace(0, 1, len(q0x)))

    for q01, q02, q03, col in zip(q0x, q0y, q0z, colors):
        q, dq = explicit_euler([q01, q02, q03], f, h, n)
        p = dq / 2
        axq.plot(q[:, 0], q[:, 1], q[:, 2], color=col, linewidth=0.125)
        axp.plot(p[:, 0], p[:, 1], p[:, 2], color=col, linewidth=0.125)
        ax1.plot(q[:, 0], p[:, 0], color=col, linewidth=0.125)
        ax2.plot(q[:, 1], p[:, 1], color=col, linewidth=0.125)
        ax3.plot(q[:, 2], p[:, 2], color=col, linewidth=0.125)
    figq.savefig("Q.png")
    figp.savefig("P.png")
    fig1.savefig("(Q1,P1).png")
    fig2.savefig("(Q2,P2).png")
    fig3.savefig("(Q3,P3).png")


# Dada la funcion que define la ecuacion diferencial,
# y los distintos t_i, generamos el espacio de fase
# asociada al cubo [-1, 1]**3
def get_phasic_space(f, t):
    I = np.linspace(-1., 1., num=5, endpoint=True)
    q0x, q0y, q0z = np.meshgrid(I, I, I)
    q0x = q0x.flatten()
    q0y = q0y.flatten()
    q0z = q0z.flatten()

    # q, _ = explicit_euler([q0x[0], q0y[0], q0z[0]], f, h, n)

    # En lugar el metodo de Euler como en apartados anteriores,
    # vamos a utilizar un metodo de una libreria ya existente
    # que resuelve ecuaciones diferenciales, para tener mas precision.
    q = odeint(f,[q0x[0], q0y[0], q0z[0]],t)
    phasic_space = q

    # Quitamos el primer punto de la malla,
    # al haber sido computado ya
    q0x = np.delete(q0x, 0)
    q0y = np.delete(q0y, 0)
    q0z = np.delete(q0z, 0)
    for q01, q02, q03 in zip(q0x, q0y, q0z):
        # q, _ = explicit_euler([q01, q02, q03], f, h, n)
        q = odeint(f, [q01, q02, q03], t)
        phasic_space = np.concatenate((phasic_space, q), axis=0)

    return phasic_space

# si ui == R =>
# sum{ui^d} = R^d*sum_i{1} = R^d * N(R)
# [N(R)] => check

# Devolvemos del valor mas cercano a x,
# entre k y k+1.
def get_index_side(kx, x, x0, h):
    if x0 + (kx + 0.5) * h >= x:
        return kx
    else:
        return kx + 1

# Dado un punto, obtenemos la tupla de
# indices que identifica al cubo mas cercano
# que lo contiene.
def get_index_subcube(p, init, h):
    idx = []
    for x, x0 in zip(p, init):
        k = int(np.floor((x - x0) / h))
        idx.append(get_index_side(k, x, x0, h))
    return np.array(idx)

# Funcion que obtiene el numero de cubos
# que contienen puntos de la superficie
def count_points(surface, init, h):
    idx_set = set([])
    for point in surface:
        idx = get_index_subcube(point, init, h)
        idx_set.add(tuple(idx))
    return len(idx_set)

# Devuelve N(V,r), dado V y r.
def n_r(r, phasic_space):
    init = [np.amin(phasic_space[:,0]), np.amin(phasic_space[:,1]), np.amin(phasic_space[:,2])]
    return count_points(phasic_space, init, r)

# DEPRECATED: No hace falta hacer busqueda,
# se puede sacar de forma constante
# Forma alternativa: para cada dimension, encontramos
# en que t del segmento pasamos de un indice de cubo a otro.
# Este metodo utiliza busqueda binaria para localizar todos estos
# puntos a la vez.
# def generate_cube_index(t0,t1,k0,k1,x0,x1,x2,h, dim):
#     # Si k0 == k1, no hay transicion entre cubos,
#     # y por tanto, devolvemos la lista vacia
#     if k0 == k1:
#         return []
#
#     # Caso base: hay transicion entre t0 y t1
#     # y estos se encuentran muy proximos entre si.
#     # Devolvemos el tiempo, la dimension, y el valor de k
#     # en el extremo derecho.
#     if t1 - t0 < eps():
#         return [(t0, dim, k1)]
#
#     # Hallamos el punto medio y generamos el indice
#     # del cubo al que pertenece
#     tmitad = (t0 + t1)/2
#     xmitad = tmitad*x2+(1-tmitad)*x1
#     kmitad = get_index_side(int(np.floor((xmitad - x0) / h)),xmitad,x0,h)
#
#     # Buscamos en las dos mitades
#     idx = generate_cube_index(t0, tmitad, k0, kmitad, x0, x1, x2, h, dim)
#     idx.extend(generate_cube_index(tmitad, t1, kmitad, k1, x0, x1, x2, h, dim))
#     return idx

# Generamos la lista de t_i de tal forma que para cada t_i
# el indice asociado a la coordenada i-esima del cubo cambia
def generate_cube_index(k1,k2,x0,x1,x2,h,dim_init):
    t_list = []

    # Hay que distinguir dos casos en funcion
    # de step. Tambien la forma de recorrer el range
    # depende.
    step = -1 if k1 < k2 else 1
    for k in reversed(range(k2,k1,step)):
        t = (x0 - x1 + (k + step * 0.5) * h) / (x2 - x1)
        t_list.append((t, dim_init, k))
    return t_list

# Funcion que dado una lista con los indices de tiempos
# para cada dimension y la tupla de indices que identifica
# al cubo inicial, devuelve un conjunto con todos los puntos
# por los que va atravesando el segmento.
def combine_idx(idx_changes, k_init):
    idx_changes = sorted(idx_changes)
    # print(idx_changes)
    i = 0
    value_set = set([])
    value_set.add(tuple(k_init))
    while i < len(idx_changes):
        t_actual, dim, k = idx_changes[i]

        # Actualizamos el valor de la dimension del
        # cubo actual, pues hemos cambiado de cubo
        k_init[dim] = k

        # Si tenemos que varios tiempos coinciden,
        # la actualizacion de indices tiene que ser a la vez.
        while i + 1 < len(idx_changes) and idx_changes[i+1][0] == t_actual:
            i += 1
            _, dim, k = idx_changes[i]
            k_init[dim] = k

        # Anyadimos el nuevo cubo generado
        value_set.add(tuple(k_init))
        i += 1
    # print(value_set)
    return value_set

# Funcion que dado dos puntos, devuelve
# todos los indices de los cubos que atraviesa
# el segmento.
def get_indixes_subcube_segment(p1, p2, init, h):
    idx_changes = []
    k_init = []
    dim_init = 0

    # Iteramos en cada dimension
    for x1, x2, x0 in zip(p1,p2, init):

        # Hallamos los indices de los extremos.
        k1 = get_index_side(int(np.floor((x1-x0) / h)),x1,x0,h)
        k2 = get_index_side(int(np.floor((x2-x0) / h)), x2, x0, h)
        # print("k1,k2=", k1, k2)
        r1 = generate_cube_index(k1, k2, x0, x1, x2, h, dim_init)
        idx_changes.extend(r1)
        dim_init += 1
        k_init.append(k1)
    return combine_idx(idx_changes, k_init)


# Funcion que devuelve N(V,r) en una version mas
# precisa que la anterior, pero mas costosa.
def n_r_1(r, phasic_space):
    init = [np.amin(phasic_space[:, 0]), np.amin(phasic_space[:, 1]), np.amin(phasic_space[:, 2])]
    index_set = set([])

    # Recorremos todos los puntos
    for i in range(1, len(phasic_space)):
        index_set |= get_indixes_subcube_segment(phasic_space[i-1], phasic_space[i], init , r)
    return len(index_set)

# Funcion que calcula hausdorff
# haciendo busqueda binaria sobre d. Utiliza
# N(l1) y N(l2) para este proposito
def hausdorff(p1, p2, i, j):
    if abs(i-j) < eps():
        return (i+j)/2
    r1, nr1 = p1
    r2, nr2 = p2
    d = (i+j)/2
    a = nr1 * (r1**d)
    b = nr2 * (r2**d)
    # print("Vamos por:", i, j, d, "y obtenemos",a, b)
    # sucesion creciente => infinito
    if eps() < b - a:
        return hausdorff(p1, p2, d, j)
    # sucesion decreciente
    elif a - b > eps():
        return hausdorff(p1, p2, i, d)
    else:
        return d


# Ejercicio 4 utilizando el metodo 1 de box-counting
def exercise4():
    a = 10
    b = 28
    c = 8 / 3
    f = lambda q,t: derivate(q, a, b, c)
    t = np.arange(0.0, 40.0, 10**-3)
    phasic_space = get_phasic_space(f,t)
    rs = [2**(-i+1) for i in range(5)]
    nr1 = n_r(2, phasic_space)
    nr = [nr1]
    hds = []
    for r in rs:
        nr2 = n_r(r/2, phasic_space)
        hd = hausdorff((r, nr1), (r/2, nr2), 1, 3)

        nr.append(nr2)
        hds.append(hd)

        print("La dimension de hausdorff para el metodo 1 y para r="
              + str(r) + " es:", "%.3f" % hd)
        nr1 = nr2

    print("Los distintos nr para el metodo 1 son : " + str(nr))

# Ejercicio 4 utilizando el metodo 2 de box-counting
def exercise_4_ext():
    a = 10
    b = 28
    c = 8 / 3
    f = lambda q, t: derivate(q, a, b, c)
    t = np.arange(0.0, 40.0, 10 ** -3)
    phasic_space = get_phasic_space(f, t)
    rs = [2 ** (-i + 1) for i in range(5)]
    nr1 = n_r_1(2, phasic_space)
    nr = [nr1]
    for r in rs:
        nr2 = n_r_1(r / 2, phasic_space)
        hd = hausdorff((r, nr1), (r / 2, nr2), 1, 3)

        nr.append(nr2)

        print("La dimension de hausdorff para el metodo 2 y para r="
              + str(r) + " es:", "%.3f" % hd)
        nr1 = nr2

    print("Los distintos nr para el metodo 2 son : " + str(nr))

if __name__ == "__main__":
    setup()
    exercise1_2()
    exercise3()

    t1_start = process_time()
    exercise4()
    t1_stop = process_time()

    print("Elapsed time during method 1 is", "%.3f" % (t1_stop - t1_start))

    t2_start = process_time()
    exercise_4_ext()
    t2_stop = process_time()

    print("Elapsed time during method 2 is", "%.3f" % (t2_stop - t2_start))


