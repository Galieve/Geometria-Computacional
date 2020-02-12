import binascii
import heapq
import os
import numpy as np
import pandas as pd
from collections import Counter


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


# Para obtener una rama del arbol de Huffman, utilizamos una cola de prioridad de tuplas (probabilidad, cadena).
# Cada vez que queremos obtener una nueva rama, cogemos las dos cadenas con menor probabilidad de la cola, le asignamos 
# a cada una un valor 0 o 1 de la rama, y metemos en la cola de prioridad la combinacion de ambas cadenas, con
# probabilidad la suma de ambas.
def huffman_branch(pq):
    prob_first, name_first = heapq.heappop(pq)
    prob_second, name_second = heapq.heappop(pq)
    heapq.heappush(pq, (prob_first + prob_second, ''.join([name_first, name_second])))
    code = np.array([{name_first: 0, name_second: 1}])
    return code, pq

# Metodo que obtiene el arbol de Huffman asociado a una distribucion. 
# Para ello, primero mete en una cola de prioridad los pares (probabilidad, caracter) de la
# distribucion, y va generando todas las ramas del arbol. Estas ramas estan ordenadas desde los nodos
# mas profundos del arbol hasta la raiz.
def huffman_tree(distr):
    tree = np.array([])
    pq = obtain_heap_from_distribution(distr)

    # Condicion de parada: solo nos queda un elemento en nuestra cola de prioridad
    while len(pq) > 1:
        code, pq = huffman_branch(pq)
        tree = np.concatenate((tree, code), axis=None)
    return tree

# Obtenemos un montículo de mínimos (o cola de prioridad) con los pares (probabilidad, caracter)
def obtain_heap_from_distribution(distr):
    lis = [(prob, name) for prob, name in zip(distr['probab'], distr['states'])]
    heapq.heapify(lis)
    return lis

# Dado una distribucion y un arbol, obtenemos la codificacion de los elementos de la distribucion 
# utilizando el arbol. Para ello, recorremos el arbol (dado en orden de hojas a raíz) y vamos
# reconstruyendo para cada caracter, la codificación asociada.
def obtain_codification_table(distr, tree):

    # Inicializamos el diccionario con la cadena vacia para todos los caracteres
    codification = {elem: '' for elem in distr['states']}

    # Recorremos todas las ramas del arbol, y las cadenas que codifican
    for branch in tree:
        for cod in branch:

            value = str(branch[cod])

            # Por cada caracter en la cadena, le añadimos por la izquierda el valor
            # 0 o 1 asociado a la codificacion
            for character in cod:
                codification[character] = ''.join([value, codification[character]])

    keys, values = list(codification.keys()), list(codification.values())

    # Guardamos en un dataframe los caracteres y la codificacion asociada
    return pd.DataFrame({'states': keys, 'code': values})


# Dada la tabla de codificacion y distribucion, devuelve la longitud media asociada
def get_average_length(cod_table, distr):
    av_l = 0
    # Recorremos la tabla de codificacion para obtener la longitud de los codigos
    for _, row in cod_table.iterrows():
        state = row["states"]
        code = row["code"]

        # Por cada caracter, obtenemos la probabilidad asociada
        prob = distr.loc[distr['states'] == state]["probab"].iloc[0]

        # Como las probabilidades ya estan normalizadas, basta con multipilicar la
        # probabilidad asociada a un caracter con la longitud de su codificacion
        av_l += + len(code) * prob
    return av_l

# Obtenemos la entropia de una distribucion, utilizando la formula de entropia de Shannon
def get_entropy(distr):
    entropy = 0
    for _, row in distr.iterrows():
        prob = row["probab"]
        entropy -= prob * np.log2(prob)
    return entropy


def exercise1(filename):
    distr = get_distribution_from_file(filename)
    tree = huffman_tree(distr)
    cod_tab = obtain_codification_table(distr, tree)
    av_length = get_average_length(cod_tab, distr)
    entropy = get_entropy(distr)
    print("If file is:", filename)
    print("The entropy equals to", entropy)
    print("And the average length =", av_length)
    if entropy <= av_length and av_length < entropy + 1:
        print("Shannon theorem holds")
    else:
        print("Shannon theorem doesn't hold! (Error)")
    print()
    return distr, tree, cod_tab

# Dado un string y la tabla de codificacion, codifica el string utilizando la tabla
def codify(cod_tab, word):
    sol = ""
    for character in word:

        # Por cada caracter, miramos en la tabla de codificacion como se codifica,
        # y le añadimos la codificacion al string solucion
        sol += cod_tab.loc[cod_tab["states"] == character]["code"].iloc[0]
    return sol

# Dado un string, devuelve la representacion binaria asociada en ASCII
def string_to_binary(word):
    return ''.join(format(ord(i), 'b') for i in word)


def exercise2(cod_tab, word):
    word_codified = codify(cod_tab, word)
    binary_word = string_to_binary(word)
    print("The word", word, "is codified as:", word_codified, "using our huffman tree, with length", len(word_codified))
    print("The word", word, "is codified as:", binary_word, "using standard codification, with length", len(binary_word))
    print("The rate of the bits used is: ", 100 * len(word_codified) / len(binary_word), "\n")


# Decodifica una palabra dada su tabla de codificacion
def decode(word, cod_tab):
    decoded_word = ""
    branch = ""

    # Recorremos caracter por caracter hasta que el string acumulado coincida con alguna
    # codificacion de algun caracter de nuestra tabla. En ese momento, reseteamos el caracter
    # acumulado y añadimos el caracter encontrado al string solucion
    for char in word:
        branch += char
        if len(cod_tab.loc[cod_tab["code"] == branch]) > 0:
            decoded_word += cod_tab.loc[cod_tab["code"] == branch]["states"].iloc[0]
            branch = ""

    # Si el string acumulador no esta vacio, quiere decir que la palabra suministrada
    # no se puede decodificar utilizando nuestra tabla, y por tanto devolvemos None
    if branch != "":
        return None
    else:
        return decoded_word


def exercise3(word, cod_tab):
    decoded_word = decode(word, cod_tab)
    if decoded_word is None:
        print("The word", word, "doesn't belong to this grammar\n")
    else:
        print("The word", word, "is decoded as:", decoded_word, "\n")


def gini_coefficient(cod_tab, distr):

    gini = 0

    # Ordenamos la distribucion por probabilidades, de tal forma que los primeros elementos
    # sean los de mayor probabilidad (y por tanto, tengan las codificaciones asociadas mas cortas)
    distr = distr.sort_values(by='probab', ascending=False)
    y_acum = 0

    # Para calcular el indice Gini y calcular el area de la curva de Lorenz, aproximamos el area
    # utilizando la regla del trapecio. Tomamos como xi las distintas probabilidades, y los yj son 
    # las longitudes acumuladas asociadas a los xi
    for _, row in distr.iterrows():

        len_code = len(cod_tab.loc[cod_tab["states"] == row["states"]]["code"].iloc[0])
        # y_j = len_code + y_acum
        # y_{j-1} = y_acum
        # x_j = x_{j-1} + row["prob"]
        gini += row["probab"] * (2 * y_acum + len_code)
        y_acum += len_code

    return 1 - gini/y_acum


def hill_diversity(distr):
    sum = 0
    for _, row in distr.iterrows():
        sum += row["probab"]**2
    return 1/sum


def exercise4(cod_tab, distr):
    print("The gini coefficient is: ", gini_coefficient(cod_tab, distr))
    print("The hill diversity is: ", hill_diversity(distr))


if __name__ == "__main__":
    setup()

    en_distr, en_tree, en_cod_tab = exercise1('auxiliar_en_pract2.txt')
    es_distr, es_tree, es_cod_tab = exercise1('auxiliar_es_pract2.txt')

    exercise2(en_cod_tab, "fractal")
    exercise2(es_cod_tab, "fractal")

    exercise3("0011111101111000010100", en_cod_tab)
    exercise4(en_cod_tab, en_distr)