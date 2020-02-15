from collections import OrderedDict
import numpy as np
import random


def f(x, r):
    return r * x * (1 - x)
# f(x) = r*x - r*x**2; f'(x) = r*(1-2*x) <= r => no contractiva


def eps():
    return 1e-5


def floyd_cycle_finding(x0, f):
    t = f(x0)
    h = f(f(x0))
    steps = 1000
    while abs(t - h) >= eps() and steps > 0:
        t = f(t)
        h = f(f(h))
        steps -= 1

    if steps == 0:
        return 0, 0

    mu = 0
    h = x0
    steps = 1000
    while abs(t - h) >= eps() and steps > 0:
        t = f(t)
        h = f(h)
        mu += 1
        steps -= 1

    lam = 1
    h = f(t)
    if steps == 0:
        return 0, 0

    steps = 1000
    while abs(t - h) >= eps() and steps > 0:
        h = f(h)
        lam += 1
        steps -= 1

    if steps == 0:
        return 0, 0
    return mu, lam


def get_partial_orbit(f, m, k, x0):
    fi = x0
    for i in range(m):
        fi = f(fi)
    orbit = [fi]
    for j in range(k - 1):
        orbit.append(f(orbit[-1]))
    return np.array(orbit)


def orbits_equivalent(o1, o2):
    if len(o1) != len(o2):
        return False
    idx = np.argmin(np.abs(o2 - o1[0]))
    for i in range(len(o1)):
        if abs(o1[i] - o2[(idx + i) % len(o1)]) >= eps():
            return False
    return True



def is_atractor(f, x0):
    m, k = floyd_cycle_finding(x0, f)
    l = get_partial_orbit(f, m, k, x0)
    for delta in np.arange(eps(), eps()/100, -eps()/100):
        md, kd = floyd_cycle_finding(x0 + delta, f)
        ld = get_partial_orbit(f, md, kd, x0)
        if not orbits_equivalent(l, ld):
            return False
    return True


def exercise1():
    l = []
    x0 = random.uniform(0, 1)
    while len(l) < 2:
        r = random.uniform(3 + eps(), 3.5 - eps())
        fr = lambda x0: f(x0, r)
        if is_atractor(fr, x0):
            m, k = floyd_cycle_finding(x0, fr)
            l.append((x0, r, get_partial_orbit(fr, m, k, x0)))
    for i in l:
        x0, r, li = i
        print("Orbit with x0:", x0, "r:", r, "set:", li)

def exercise2():

    x0 = random.uniform(0, 1)
    orb = None
    found = False
    while not found:
        r = random.uniform(3 + eps(), 4 - eps())
        fr = lambda x0: f(x0, r)
        if is_atractor(fr, x0):
            m, k = floyd_cycle_finding(x0, fr)
            orb = get_partial_orbit(fr, m, k, x0)
            if len(orb) == 8:
                found = True

    print("Orbit with x0:", x0, "r:", r, "set:", orb)


if __name__ == "__main__":

  exercise1()
  exercise2()
