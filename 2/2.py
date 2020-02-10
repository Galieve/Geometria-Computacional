"""
Practica 2
"""
import binascii
import heapq
import os

import numpy as np
import pandas as pd
from collections import Counter


def setup():
    #### Carpeta donde se encuentran los archivos ####
    ubica = os.path.dirname(os.path.abspath(__file__))

    #### Vamos al directorio de trabajo####
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


## Ahora definimos una funcion que haga exactamente lo mismo
# pq = priority_queue de {prod, estad} ordenados por probabilidad
def huffman_branch(pq):
    prob_first, name_first = heapq.heappop(pq)
    prob_second, name_second = heapq.heappop(pq)
    heapq.heappush(pq, (prob_first + prob_second, ''.join([name_first, name_second])))
    code = np.array([{name_first: 0, name_second: 1}])
    return code, pq


def huffman_tree(distr):
    tree = np.array([])
    pq = obtain_heap_from_distribution(distr)
    while len(pq) > 1:
        code, pq = huffman_branch(pq)
        tree = np.concatenate((tree, code), axis=None)
    return tree


def obtain_heap_from_distribution(distr):
    lis = [(prob, name) for prob, name in zip(distr['probab'], distr['states'])]
    heapq.heapify(lis)
    return lis


def obtain_codification_table(distr, tree):
    codification = {elem: '' for elem in distr['states']}
    for branch in tree:
        for cod in branch:
            value = str(branch[cod])
            for character in cod:
                codification[character] = ''.join([value, codification[character]])
    keys, values = list(codification.keys()), list(codification.values())
    return pd.DataFrame({'states': keys, 'code': values})


# estan los pesos normalizados, asi que W = 1.
def get_average_length(cod_table, distr):
    av_l = 0
    for _, row in cod_table.iterrows():
        state = row["states"]
        code = row["code"]
        prob = distr.loc[distr['states'] == state]["probab"].iloc[0]
        av_l += + len(code) * prob
    return av_l


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
    print("file =", filename)
    print("entropy =", entropy)
    print("average length =", av_length)
    if entropy <= av_length and av_length < entropy + 1:
        print("Shannon theorem holds")
    else:
        print("Shannon theorem doesn't hold! (Error)")
    print()
    return distr, tree, cod_tab


def codify(cod_tab, word):
    sol = ""
    for character in word:
        sol += cod_tab.loc[cod_tab["states"] == character]["code"].iloc[0]
    return sol


def string_to_binary(word):
    return ''.join(format(ord(i), 'b') for i in 'fractal')


def exercise2(cod_tab, word):
    word_codified = codify(cod_tab, word)
    binary_word = string_to_binary(word)
    print("the word", word, "is codified as:", word_codified, "using our huffman tree")
    print("the word", word, "is codified as:", binary_word, "using standard codification")
    print("the rate of the bits used is: ", 100 * len(word_codified) / len(binary_word), "\n")


def decode(word, cod_tab):
    decoded_word = ""
    branch = ""
    for char in word:
        branch += char
        if len(cod_tab.loc[cod_tab["code"] == branch]) > 0:
            decoded_word += cod_tab.loc[cod_tab["code"] == branch]["states"].iloc[0]
            branch = ""

    if branch != "":
        return None
    else:
        return decoded_word


def exercise3(word, cod_tab):
    decoded_word = decode(word, cod_tab)
    if decoded_word is None:
        print("The word", word, "doesn't belong to this grammar\n")
    else:
        print(decoded_word, "\n")


def gini_coefficient(cod_tab, distr):

    gini = 0

    distr = distr.sort_values(by='probab', ascending=False)
    y_acum = 0
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
    print("the gini coefficient is: ", gini_coefficient(cod_tab, distr))
    print("the hill diversity is: ", hill_diversity(distr))


if __name__ == "__main__":
    setup()

    en_distr, en_tree, en_cod_tab = exercise1('auxiliar_en_pract2.txt')
    es_distr, es_tree, es_cod_tab = exercise1('auxiliar_es_pract2.txt')

    exercise2(en_cod_tab, "fractal")
    exercise2(es_cod_tab, "fractal")


    exercise3("0000111011111111111010", en_cod_tab)
    exercise3("1111000100000000000101", en_cod_tab)

    exercise4(en_cod_tab, en_distr)
