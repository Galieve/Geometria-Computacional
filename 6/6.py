
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz

def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)

#q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parámetro temporal
def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq

#Ecuación de un sistema dinámico continuo
#Ejemplo de oscilador simple
def F(q):
    ddq = - 2*q*(q**2 - 1)
    return ddq

#Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
#Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n,q0,dq0,F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q #np.array(q),


def periodos(q,d,max=True):
    #Si max = True, tomamos las ondas a partir de los máximos/picos
    #Si max == False, tomamos los las ondas a partir de los mínimos/valles
    epsilon = 6*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante
    avg_point = sum(q) / q.size
    if max:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q > avg_point))
    else:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q < avg_point))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves

## Pintamos el espacio de fases
def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/10**(-4)),marker='-'):
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col))


def exercise1():
    d = 10**(-3.5)
    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    seq_q0 = np.linspace(0., 1., num=10, endpoint=True)
    seq_dq0 = np.linspace(0., 2, num=10, endpoint=True)
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            # ax = fig.add_subplot(1, 1, 1)
            col = (1 + i + j * (len(seq_q0))) / (len(seq_q0) * len(seq_dq0))
            # ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
            simplectica(q0=q0, dq0=dq0, F=F, col=col, marker='-', d=d, n=int(16 / d))
    ax = fig.gca()
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    fig.savefig('Simplectic.png', dpi=250)
    plt.show()


def find_orbit_area(q0, dq0, d, n):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq / 2

    T, W = periodos(q, d, max=False)
    print(W)
    if len(W) < 2:
        return 0, True
    mitad = np.arange(W[0], W[0] + np.int((W[1] - W[0]) / 2), 1)
    print(p[mitad[-1]])
    area = simps(q[mitad], p[mitad])
    return 2 * area, np.min(q) < 0

def find_area(seq_q0, seq_dq0, d, n):
    maxim = 0
    peque = 100
    minim = 100
    q1 = 0
    dq1 = 0
    for q0, dq0 in zip(seq_q0, seq_dq0):
        area, cond = find_orbit_area(q0,dq0,d,n)
        if area == 0:
            continue
        if cond:
            if peque > area:
                q1 = q0
                dq1 = dq0
            peque = min(area, peque)
        minim = min(minim, area)
        maxim = max(maxim, area)
    print(q1, dq1)
    return maxim - peque / 2 - minim

def check_liouville_theorem():
    d = 10**(-3)
    n = int(10 / d)
    seq_q0 = np.linspace(0., 1., num=10, endpoint=True)
    seq_dq0 = np.linspace(0., 2, num=10, endpoint=True)
    seq_dq0, seq_q0 = np.meshgrid(seq_dq0, seq_q0)
    seq_q0 = seq_q0.flatten()
    seq_dq0 = seq_dq0.flatten()
    t_list = np.asarray(np.arange(0, 6) / d, dtype=int)
    list_seq_qi = []
    list_seq_dqi = []
    for q0, dq0 in zip(seq_q0, seq_dq0):
        q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
        dq = deriv(q, dq0=dq0, d=d)
        list_seq_qi.append(q[t_list])
        list_seq_dqi.append(dq[t_list])
    list_seq_qi = np.asarray(list_seq_qi).transpose()
    list_seq_dqi = np.asarray(list_seq_dqi).transpose()
    areas = []
    n = int(32 / d)
    for seq_qi, seq_dqi in zip(list_seq_qi, list_seq_dqi):
        # plt.scatter(seq_qi, seq_dqi)
        # plt.show()
        # plt.clf()

        # print("HOLA")
        areas.append(find_area(seq_qi, seq_dqi, d, n))
        print(areas[-1])
    print(areas)





def exercise2():
    deltas = np.linspace(10**-3., 10**-4., num=10)
    deltas = [0.001, 0.002]
    areas = []
    for d in deltas:
        n = int(32/d)
        print(d)
        seq_q0 = np.linspace(0., 1., num=10, endpoint=True)
        seq_dq0 = np.linspace(0., 2, num=10, endpoint=True)
        seq_dq0, seq_q0 = np.meshgrid(seq_dq0, seq_q0)
        seq_q0 = seq_q0.flatten()
        seq_dq0 = seq_dq0.flatten()
        areas.append(find_area(seq_q0, seq_dq0,d,n))
    diff_areas = np.diff(np.asarray(areas))
    print("El area estimada del espacio fásico es " + str(areas[0]) +
         " y el error cometido es " + str(max(abs(diff_areas))))




if __name__ == "__main__":
    setup()
    print(find_orbit_area(0.222222, 0.22222, 10**-3, int(48/(10**-3))))
    simplectica(0.22222, 0.22222, F, d = 10**-3, n = int(48/10**(-3)))
    plt.show()
    # check_liouville_theorem()
    # exercise1()
    # exercise2()
    # seq_q0 = np.linspace(0., 1., num=10)
    # seq_dq0 = np.linspace(0., 2, num=10)
    # print(find_area(seq_q0, seq_dq0, 0.0002, int(32/0.0002)))
    # print(find_orbit_area(0.222222,2.0, 0.0002, int(32/0.0002)))