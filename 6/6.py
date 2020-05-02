# Practica realizada por Alejandro Hernandez 
# y Enrique Roman Calvo

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

def setup():
    workpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workpath)

#q = variable de posicion, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parametro temporal
def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq

#Ecuacion de un sistema dinamico continuo
#Ejemplo de oscilador simple
def F(q):
    ddq = - 2*q*(q**2 - 1)
    return ddq

#Resolucion de la ecuacion dinamica \ddot{q} = F(q), obteniendo la orbita q(t)
#Los valores iniciales son la posicion q0 := q(0) y la derivada dq0 := \dot{q}(0)
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
    #Si max == True, tomamos las ondas a partir de los maximos/picos
    #Si max == False, tomamos los las ondas a partir de los minimos/valles
    epsilon = 6*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante

    # Para comprobar si es un pico o un valle, comparamos
    # el valor de q con la media de los valores de q (en lugar de compararlo
    # con 0) Si es < 0, es un valle, y si es > 0, es un pico
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
    # Iteramos en la secuencia de puntos iniciales
    # y cada orbita asociada
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            col = (1 + i + j * (len(seq_q0))) / (len(seq_q0) * len(seq_dq0))
            simplectica(q0=q0, dq0=dq0, F=F, col=col, marker='-', d=d, n=int(16 / d))
    ax = fig.gca()
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    fig.savefig('Simplectic.png', dpi=250)
    plt.show()

# Funcion que calcula el area encerrada
# por la orbita definida por los puntos iniciales
# q0 y dq0
def find_orbit_area(q0, dq0, d, n):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq / 2

    T, W = periodos(q, d, max=False)

    # Si no hemos encontrado dos valles, entonces
    # no hemos conseguido encontrar una orbita estable,
    # y por tanto, devolvemos que el area es 0.
    if len(W) < 2:
        return 0
    mitad = np.arange(W[0], W[0] + np.int((W[1] - W[0]) / 2), 1)

    # Aplicamos la regla de Simpson para calcular
    # el area
    area = simps(p[mitad], q[mitad])
    return 2 * area

# Encuentra el area del espacio fasico D.
# Notese que esta funcion tiene en cuenta
# la figura que hemos obtenido del espacio fasico
# en el primer apartado para calcular el area.
# La sequencia de q0 y dq0 son arrays 1-d que representan
# la malla de puntos
def find_area(seq_q0, seq_dq0, d, n):

    # El area del espacio fasico la vamos a calcular como
    # el area de la orbita mas externa (la que tiene mayor area)
    # menos el area de la orbita de menor area centrada en el punto
    # (1,0), menos la mitad del area de la orbita mas pequeña
    # que tiene forma parecida al simbolo de infinito (para asi quitar
    # el hueco que queda sin cubrir, que se aproxima a ese valor)
    maxim = 0
    peque = 0
    minim = 100

    for q0, dq0 in zip(seq_q0, seq_dq0):
        area = find_orbit_area(q0,dq0,d,n)
        if area == 0:
            continue
        # La primera orbita con area distinta de cero corresponde
        # con la orbita mas pequeña parecida a infinito, pues empezamos
        # a recorrer por q0 = 0, dq0 = 0 y vamos ascendiendo sobre dq0.
        if peque == 0:
            peque = area
        minim = min(minim, area)
        maxim = max(maxim, area)
    return maxim - peque / 2 - minim

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


# Vamos a comprobar que el teorema de Liouville se cumple para
# la distribucion de fases en distintos tiempos
def check_liouville_theorem():
    d = 10 ** (-3)
    n = int(24 / d)
    n_points = 200

    # Generamos los puntos de los lados del rectangulo
    # [0,1] x [0,2] con linspace y meshgrid

    seq_arr = np.linspace(0., 1., num=n_points, endpoint=False)
    seq_iz = np.linspace(0., 2, num=n_points, endpoint=False)
    seq_ab = np.linspace(1., 0., num=n_points, endpoint=False)
    seq_der = np.linspace(2., 0, num=n_points, endpoint=False)

    seq_q1, seq_dq1 = np.meshgrid(seq_arr, np.asarray(2))
    seq_q2, seq_dq2 = np.meshgrid(np.asarray(1), seq_der)
    seq_q3, seq_dq3 = np.meshgrid(seq_ab, np.asarray(0))
    seq_q4, seq_dq4 = np.meshgrid(np.asarray(0), seq_iz)

    seq_q0 = np.concatenate((seq_q1, seq_q2, seq_q3, seq_q4), axis=None)
    seq_dq0 = np.concatenate((seq_dq1, seq_dq2, seq_dq3, seq_dq4), axis=None)

    # Vamos a comprobar si se cumple el teorema para los tiempos de t=0 a t=15
    timestamps = np.arange(0,16)
    t_list = np.asarray(timestamps / d, dtype=int)
    list_seq_qi = []
    list_seq_pi = []

    # Vemos como se transforman los puntos del rectangulo.
    # Para ello, generamos la orbita q y el valor p de cada uno de ellos,
    # y nos quedamos con los indices correspondientes a los tiempos que queremos
    # estudiar
    for q0, dq0 in zip(seq_q0, seq_dq0):
        q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
        dq = deriv(q, dq0=dq0, d=d)
        p = dq/2
        list_seq_qi.append(q[t_list])
        list_seq_pi.append(p[t_list])

    # Tal y como hemos generado los puntos, tenemos que transponer los
    # vectores para obtener como evolucionan todos los puntos de los extremos
    # para los distintos tiempos
    list_seq_qi = np.asarray(list_seq_qi).transpose()
    list_seq_dqi = np.asarray(list_seq_pi).transpose()

    # Con esos puntos, obtenemos el area del poligono que definen
    for t, seq_qi, seq_dqi in zip(timestamps, list_seq_qi, list_seq_dqi):
        print("Area for D(t=" + str(t) + "): " + str(poly_area(seq_qi, seq_dqi)))

def exercise2():
    deltas = np.linspace(10**-3., 10**-4., num=10)
    areas = []
    for d in deltas:
        n = int(32/d)
        seq_q0 = np.linspace(0., 1., num=10, endpoint=True)
        seq_dq0 = np.linspace(0., 2, num=10, endpoint=True)
        seq_dq0, seq_q0 = np.meshgrid(seq_dq0, seq_q0)
        seq_q0 = seq_q0.flatten()
        seq_dq0 = seq_dq0.flatten()
        areas.append(find_area(seq_q0, seq_dq0,d,n))
    diff_areas = np.diff(np.asarray(areas))
    print("El area estimada del espacio fasico es " + str(areas[0]) +
         " y el error cometido es " + str(max(abs(diff_areas))))
    check_liouville_theorem()




if __name__ == "__main__":
    setup()
    exercise1()
    exercise2()