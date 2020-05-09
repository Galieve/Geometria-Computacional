import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from scipy.integrate import simps
from skimage import io, color
from matplotlib import animation


def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


def eps():
    return 10**-5


def derivate(q, a, b, c):
    q1, q2, q3 = q
    r1 = a * (q2 - q1)
    r2 = b * q1 - q2 - q1 * q3
    r3 = q1 * q2 - c * q3
    return [r1, r2, r3]


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


def get_phasic_space(f, t):
    I = np.linspace(-1., 1., num=5, endpoint=True)
    q0x, q0y, q0z = np.meshgrid(I, I, I)
    q0x = q0x.flatten()
    q0y = q0y.flatten()
    q0z = q0z.flatten()

    # q, _ = explicit_euler([q0x[0], q0y[0], q0z[0]], f, h, n)
    q = odeint(f,[q0x[0], q0y[0], q0z[0]],t)
    phasic_space = q
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


def get_index_side(kx, x, x0, h):
    if x0 + (kx + 0.5) * h >= x:
        return kx
    else:
        return kx + 1


def get_index_subcube(p, init, h):
    idx = []
    for x, x0 in zip(p, init):
        k = int(np.floor((x - x0) / h))
        idx.append(get_index_side(k, x, x0, h))
    return np.array(idx)


def count_points(surface, init, h):
    idx_set = set([])
    for p in surface:
        idx = get_index_subcube(p, init, h)
        idx_set.add(tuple(idx))
    return len(idx_set)


def n_r(r, phasic_space):
    init = [np.amin(phasic_space[:,0]), np.amin(phasic_space[:,1]), np.amin(phasic_space[:,2])]
    return count_points(phasic_space, init, r)


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



def exercise4():
    a = 10
    b = 28
    c = 8 / 3
    f = lambda q,t: derivate(q, a, b, c)
    t = np.arange(0.0, 40.0, 10**-4)
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

        print("La dimension de hausdorff para r=" + str(r) + " es: " + str(hd))
        nr1 = nr2

    print("Los distintos nr son : " + str(nr))


if __name__ == "__main__":
    setup()
    exercise1_2()
    exercise3()
    exercise4()
