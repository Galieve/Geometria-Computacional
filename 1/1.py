# Enrique Roman Calvo y Alejandro Hernandez Cerezo

import numpy as np
import random
from matplotlib import pyplot as plt


# Definimos la funcion logistica, que depende tanto de
# r como de x.
def f(x, r):
    return r * x * (1 - x)

# f(x) = r*x - r*x**2; |f'(x)| = |r*(1-2*x)| <= r => no contractiva (siempre)


# Por comodidad, definimos un epsilon constante
def eps():
    return 1e-5

# Algoritmo de floyd para detectar ciclos. Tiene como tope
# un maximo de 10.000 iteraciones
def floyd_cycle_finding(x0, f):
    t = f(x0)
    h = f(f(x0))
    steps = 10000

    # Determinamos el punto en el que empieza el ciclo
    while abs(t - h) >= eps() and steps > 0:
        t = f(t)
        h = f(f(h))
        steps -= 1

    if steps == 0:
        return 0, 0

    mu = 0
    h = x0
    steps = 10000

    # Hallamos la longitud del ciclo
    while abs(t - h) >= eps() and steps > 0:
        t = f(t)
        h = f(h)
        mu += 1
        steps -= 1

    lam = 1
    h = f(t)
    if steps == 0:
        return 0, 0

    steps = 10000

    # Hallamos el valor de M (iteraciones hasta encontrar
    # el ciclo)
    while abs(t - h) >= eps() and steps > 0:
        h = f(h)
        lam += 1
        steps -= 1

    if steps == 0:
        return 0, 0
    return mu, lam

# Para hallar la orbita parcial dado el m y el k,
# iteramos hasta x_m y los siguientes k valores forman la orbita
def get_partial_orbit(f, m, k, x0):
    fi = x0
    for i in range(m):
        fi = f(fi)
    orbit = [fi]
    for j in range(k - 1):
        orbit.append(f(orbit[-1]))
    return np.array(orbit)

# Para ver si dos orbitas son equivalentes, encuentro el indice del termino de la
# segunda orbita que corresponde con la primera, y a partir de ahi itero para ver si
# todos los valores coinciden
def orbits_equivalent(o1, o2):
    if len(o1) != len(o2):
        return False
    idx = np.argmin(np.abs(o2 - o1[0]))
    for i in range(len(o1)):
        if abs(o1[i] - o2[(idx + i) % len(o1)]) >= eps():
            return False
    return True

# Dado x0 y f, hallo el conjunto límite, y compruebo si hay algun punto lo suficientemente cercano
# con el mismo conjunto limite.
def is_atractor(f, x0):
    m, k = floyd_cycle_finding(x0, f)
    l = get_partial_orbit(f, m, k, x0)
    for delta in np.arange(eps(), eps() / 100, -eps() / 100):
        md, kd = floyd_cycle_finding(x0 + delta, f)
        ld = get_partial_orbit(f, md, kd, x0)
        if not orbits_equivalent(l, ld):
            return False
    return True

# Pintamos una funcion que representan las distintas iteraciones de f(x)
# hasta encontrar el conjunto limite. El punto inicial y los puntos del conjunto
# limite se marcan en el plano
def plot_function(fr, x0, m, k, number):
    plt.clf()
    orb = get_partial_orbit(fr, 0, m + k, x0)
    x = orb[: len(orb) - 1]
    y = orb[1:]

    plt.plot(x, y, color='teal')

    plt.plot(x[0], y[0], 'o', color='goldenrod')

    atractor = get_partial_orbit(fr, m, k, x0)
    x = atractor
    y = atractor[1:]
    y = np.append(y, fr(x[-1]))
    plt.scatter(x, y, color='violet')

    plt.xlabel('x')
    plt.ylabel('x')

    plt.savefig('orbit' + str(number) + '.png')

# Pintar los conjuntos atractores para distintos valores de r
def plot_attraction_graphic():
    plt.clf()
    r_range = np.arange(2.5, 4, 0.01)
    x_range = [0.5]
    plt.xlabel('r')
    plt.ylabel('x')

    for x0 in x_range:
        for r in r_range:
            fr = lambda x0: f(x0,r)
            m, k = floyd_cycle_finding(x0, fr)
            orbit = get_partial_orbit(fr, m, k, x0)
            if m != 0 or k != 0:
                plt.scatter([r] * len(orbit), orbit)
            else:
                print("Punto repulsor con x0=", x0, "y r=", r)
    plt.savefig('attraction.png')

# Dado dos conjuntos s0 y s1, hallamos el error entre ambos
# como el maximo de las diferencias entre las parejas de elementos
def subset_error(s0, s1):
    max_val = 0
    for el1, el2 in zip(s0, s1):
        max_val = max([max_val, abs(el1-el2)])
    return max_val

def calcular_error(x0, fr, m, k):
    val = x0
    for _ in range(m):
        val = fr(val)
    s1 = []
    for _ in range(k):
        s1.append(val)
        val = fr(val)
    error = 0
    for _ in range(10):
        s2 = []
        for _ in range(k):
            s2.append(val)
            val = fr(val)
        error = max([error, subset_error(s1, s2)])
        s1 = s2
    return error

def exercise1():
    l = []
    # Fijo el mismo x0 para las 2 orbitas que quiero encontrar.
    # Como no esta especificado, x0 es un valor aleatorio entre 0 y 1
    x0 = random.uniform(0, 1)

    # Tengo que encontrar 2 orbitas distintas
    while len(l) < 2:
        r = random.uniform(3 + eps(), 3.5 - eps())
        fr = lambda x0: f(x0, r)
        if is_atractor(fr, x0):
            m, k = floyd_cycle_finding(x0, fr)
            orbit = get_partial_orbit(fr, m, k, x0)
            if len(l) == 0:
                l.append((x0, r, orbit, m, k))
            # Si ya tengo una orbita calculada, compruebo que la obtenida no coincide
            elif len(l) == 1 and not orbits_equivalent(l[0], orbit):
                l.append((x0, r, orbit, m , k))
    for j,i in enumerate(l):
        x0, r, li, m, k = i
        print("Orbit with r =", r, "x0 =", x0, "set=", li, "with error:", calcular_error(x0,fr,m,k))
        # plot_function(fr, x0, m, k, j)

def exercise2():
    x0 = random.uniform(0, 1)
    orb = None
    found = False
    while not found:

        # Obtengo un valor aleatorio de r en cada iteracion
        r = random.uniform(3 + eps(), 4 - eps())
        fr = lambda x0: f(x0, r)
        if is_atractor(fr, x0):
            m, k = floyd_cycle_finding(x0, fr)
            orb = get_partial_orbit(fr, m, k, x0)
            if len(orb) == 8:
                found = True

    print("Orbit with r:", r, "x0:", x0, "set:", orb, "with error:", calcular_error(x0,fr,m,k))
    # plot_function(fr, x0, m, k, 2)


if __name__ == "__main__":
    exercise1()
    exercise2()
    plot_attraction_graphic()

