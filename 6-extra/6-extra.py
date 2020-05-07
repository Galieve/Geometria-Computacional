import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from scipy.integrate import simps
from skimage import io, color
from matplotlib import animation


def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)


# se puede realizar la aproximacion de ANNU siguiente:
# q_i(t+h)=q_i(t)+h*(dq_i/dt(t)). Además, no tiene sentido buscar
# algo como en la practica 6 ya que el error cometido es del mismo
# orden y es mucho más comodo calcular los q_i así.
# (Euler explícito).
# Podríamos hacer el trapecio o runge kutta si te hace ilu :)

# d2q/dt2 = (d2q_1/dt2, d2q_2/dt2, d2q_3/dt2) =
#   (a(dq2/dt-dq1/dt), bdq1/dt -dq2/dt -(dq1/dtq3+q1dq3/dt),
#       dq1/dtq2 + q1dq2/dt- cdq3/dt) = F(dq/dt)
# F(dq/dt) =
#   a(bq1-q2-q1q3-a(q2-q1))
#   b(a(q2-q1))-(bq1-q2-q1q3)-a(q2-q1)*q3 -q1*(q1q2-cq3)
#   a(q2-q1)*q2 +q1*(bq1-q2-q1q3) - c*(q1q2-cq3)

# dq/dt = (dq1/dt, dq2/dt, dq3/dt) \aprox = (q1(t+h)-q1(t), ...) / h

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


# si ui == R =>
# sum{ui^d} = R^d*sum_i{1} = R^d * N(R)
# [N(R)] => check

def get_points(phasic_space, r):
    xmin, xmax = np.amin(phasic_space[:, 0]), np.amax(phasic_space[:, 0])
    ymin, ymax = np.amin(phasic_space[:, 1]), np.amax(phasic_space[:, 1])
    zmin, zmax = np.amin(phasic_space[:, 2]), np.amax(phasic_space[:, 2])

    xmaxr = int(np.ceil((xmax - xmin) / r)) * r + xmin
    ymaxr = int(np.ceil((ymax - ymin) / r)) * r + ymin
    zmaxr = int(np.ceil((zmax - zmin) / r)) * r + zmin

    Ix = np.linspace(xmin, xmaxr, num=(xmaxr - xmin) // r, endpoint=True)
    Iy = np.linspace(ymin, ymaxr, num=(ymaxr - ymin) // r, endpoint=True)
    Iz = np.linspace(zmin, zmaxr, num=(zmaxr - zmin) // r, endpoint=True)

    X, Y, Z = np.meshgrid(Ix, Iy, Iz)
    #coord = {X[i],Y[i],Z[i]: for i in range(len(X))}
    coord = {str(i): Xi for i, Xi in enumerate([X.flatten(), Y.flatten(), Z.flatten()])}
    df = pd.DataFrame.from_dict(coord)
    points = df.values
    return points


def count_points(points, surface, d):
    Ix = 
    count = 0
    for p in points:
        if np.any(np.linalg.norm(surface - p, np.inf) < d):
            count += 1
    return count


def n_r(r, f, h, n):
    I = np.linspace(-1., 1., num=5, endpoint=True)
    q0x, q0y, q0z = np.meshgrid(I, I, I)
    q0x = q0x.flatten()
    q0y = q0y.flatten()
    q0z = q0z.flatten()

    q, _ = explicit_euler([q0x[0], q0y[0], q0z[0]], f, h, n)
    phasic_space = q
    q0x = np.delete(q0x, 0)
    q0y = np.delete(q0y, 0)
    q0z = np.delete(q0z, 0)
    for q01, q02, q03 in zip(q0x, q0y, q0z):
        q, _ = explicit_euler([q01, q02, q03], f, h, n)
        phasic_space = np.concatenate((phasic_space, q), axis=0)

    points_r = get_points(phasic_space, r)
    c1 = count_points(points_r, phasic_space, r/2)
    points_r = get_points(phasic_space, r/2)
    c2 = count_points(points_r, phasic_space, r/4)
    return np.array([c1, c2])


if __name__ == "__main__":
    setup()
    a = 10
    b = 28
    c = 8 / 3
    f = lambda q: derivate(q, a, b, c)
    h = 10 ** -3
    n = int(32 / h)
    # exercise1_2()
    # exercise3()
    eneerre = n_r(1, f, h, n)
    print(eneerre)
