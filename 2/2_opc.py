# Practica realizada por Alejandro Hernandez Cerezo y Enrique Roman Calvo

import heapq
import os
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from scipy import interpolate

def setup():
    # Carpeta donde se encuentran los archivos
    ubica = os.path.dirname(os.path.abspath(__file__))

    # Vamos al directorio de trabajo
    os.getcwd()
    os.chdir(ubica)


def get_distribution_from_file(filename):
    with open(filename, 'r') as file:
        msj = file.read()

    #### Pasamos todas las letras a minusculas
    msj = msj.lower()

    #### Contamos cuantas letras hay en cada texto
    tab = Counter(msj)
    tab_states = np.array(list(tab))
    tab_weights = np.array(list(tab.values()))
    tab_probab = tab_weights / float(np.sum(tab_weights))
    distr = pd.DataFrame({'states': tab_states, 'probab': tab_probab})
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index = np.arange(0, len(tab_states))

    return distr


# Para obtener una rama del arbol de Huffman, utilizamos una cola de prioridad
# de tuplas (probab., cadena). Cada vez que queremos obtener una nueva rama,
# cogemos las dos cadenas con menor probabilidad de la cola, le asignamos
# a cada una un valor 0 o 1 de la rama, y metemos en la cola de prioridad
# la combinacion de ambas cadenas, con probabilidad la suma de ambas.
def huffman_branch(pq):
    prob_first, name_first = heapq.heappop(pq)
    prob_second, name_second = heapq.heappop(pq)
    heapq.heappush(pq, (prob_first + prob_second,
                        ''.join([name_first, name_second])))
    code = np.array([{name_first: 0, name_second: 1}])
    return code, pq


# Metodo que obtiene el arbol de Huffman asociado a una distribucion.
# Para ello, primero mete en una cola de prioridad los pares
# (probabilidad, caracter) de la distribucion, y va generando todas las ramas
# del arbol Estas ramas estan ordenadas desde los nodos mas profundos
# del arbol hasta la raiz.
def huffman_tree(distr):
    tree = np.array([])
    pq = obtain_heap_from_distribution(distr)

    # Condicion de parada:
    # solo nos queda un elemento en la cola de prioridad
    while len(pq) > 1:
        code, pq = huffman_branch(pq)
        tree = np.concatenate((tree, code), axis=None)
    return tree


# Obtenemos un monticulo de minimos (o cola de prioridad)
# con los pares (probabilidad, caracter)
def obtain_heap_from_distribution(distr):
    lis = [(prob, name) for prob, name in zip(distr['probab'], distr['states'])]
    heapq.heapify(lis)
    return lis


# Dado una distribucion y un arbol, obtenemos la codificacion de los elementos
# de la distribucion utilizando el arbol. Para ello, recorremos el arbol
# desde las hojas a hasta la raiz y vamos reconstruyendo para cada caracter,
# la codificacion asociada.
def obtain_codification_table(distr, tree):
    # Inicializamos el diccionario con la cadena vacia para todos los caracteres
    codification = {elem: '' for elem in distr['states']}

    # Recorremos todas las ramas del arbol, y las cadenas que codifican
    for branch in tree:
        for cod in branch:

            value = str(branch[cod])

            # Por cada caracter en la cadena, le añadimos por la izquierda
            # el valor 0 o 1 asociado a la codificacion
            for character in cod:
                codification[character] = \
                    ''.join([value, codification[character]])

    keys, values = list(codification.keys()), list(codification.values())

    # Guardamos en un dataframe los caracteres y la codificacion asociada
    return pd.DataFrame({'states': keys, 'code': values})

def exercise1(filename):
    distr = get_distribution_from_file(filename)
    tree = huffman_tree(distr)
    cod_tab = obtain_codification_table(distr, tree)
    return distr, tree, cod_tab

def gini_coefficient_using_trapezoidal_rule(cod_tab, distr):
    gini = 0

    # Ordenamos la distribucion por probabilidades, de tal forma que
    # los primeros elementos sean los de mayor probabilidad, y por tanto,
    # tengan las codificaciones asociadas mas cortas)
    distr = distr.sort_values(by='probab', ascending=False)
    y_acum = 0

    # Para calcular el indice Gini y calcular el area de la curva de Lorenz,
    # aproximamos el area utilizando la regla del trapecio. Tomamos como xi
    # las distintas probabilidades, y los yj son
    # las longitudes acumuladas asociadas a los xi
    for _, row in distr.iterrows():
        len_code = len(cod_tab.loc[cod_tab["states"] == row["states"]]
                       ["code"].iloc[0])
        # y_j = len_code + y_acum
        # y_{j-1} = y_acum
        # x_j = x_{j-1} + row["prob"]
        gini += row["probab"] * (2 * y_acum + len_code)
        y_acum += len_code

    return 1 - gini / y_acum

# Dada la tabla de codificacion y la distribucion, obtenemos los arrays de
# frecuencias acumuladas, y longitudes de caracter acumuladas.
def obtain_lorenz_points(cod_tab, distr):
    gini = 0

    # Ordenamos la distribucion por probabilidades, de tal forma que
    # los primeros elementos sean los de mayor probabilidad, y por tanto,
    # tengan las codificaciones asociadas mas cortas)
    distr = distr.sort_values(by='probab', ascending=False)

    x_acum = 0
    y_acum = 0

    x = np.zeros(len(distr) + 1)
    y = np.zeros(len(distr) + 1)

    x[0] = 0
    y[0] = 0

    for i, tup in enumerate(distr.iterrows()):
        _, row = tup
        x_acum += row["probab"]
        x[i+1] = x_acum

        len_code = len(cod_tab.loc[cod_tab["states"] == row["states"]]
                       ["code"].iloc[0])

        y_acum += len_code
        y[i+1] = y_acum

    y /= y_acum
    return x, y


def hill_diversity(distr):
    sum = 0
    for _, row in distr.iterrows():
        sum += row["probab"] ** 2
    return 1 / sum

def exercise4(cod_tab, distr, name):
    print("For", name, "distr:")
    print("The gini coefficient using trapezoidal rule is:", gini_coefficient_using_trapezoidal_rule(cod_tab, distr))
    print("The gini coefficient using interpolation is:", gini_coefficient_using_interpolation(cod_tab, distr))
    hill = hill_diversity(distr)
    print("The hill diversity is:", hill)
    print("The Simpson index is:", 1/hill)
    print("Number of chars: ", len(distr))
    print("Inverse of number of chars: ", 1 / len(distr))
    print()


def gini_coefficient_using_interpolation(cod_tab, distr):
    x, y = obtain_lorenz_points(cod_tab, distr)
    f = interpolate.InterpolatedUnivariateSpline(x, y)


    return 1-2*f.integral(0,1)

def plot_lauren_curve(cod_tab, distr, name):
    plt.clf()
    x, y = obtain_lorenz_points(cod_tab, distr)
    f = interpolate.InterpolatedUnivariateSpline(x,y)
    xs = np.linspace(0.0,1.0, num=100)
    plt.plot(xs, f(xs), 'r')
    plt.plot(xs, xs, 'b')
    plt.scatter(x, y, c='g')
    plt.savefig(name)


if __name__ == "__main__":
    setup()

    en_distr, _, en_cod_tab = exercise1('auxiliar_en_pract2.txt')
    es_distr, _, es_cod_tab = exercise1('auxiliar_es_pract2.txt')

    exercise4(en_cod_tab, en_distr, "english")
    exercise4(es_cod_tab, es_distr, "spanish")

    # plot_lauren_curve(en_cod_tab, en_distr, 'eng_lauren.png')
    # plot_lauren_curve(es_cod_tab, es_distr, 'span_lauren.png')