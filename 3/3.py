# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:14:02 2020

@author: usu321
"""
import matplotlib
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


# Obtenemos una muestra de 1000 puntos centrados en (1,1), (-1,-1), (1, -1) con
# una desviación de 0.4 respecto a ellos.
def obtain_sample():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers,
                                cluster_std=0.4, random_state=0)
    return X, labels_true


# Dado un rango de valores de vecindades (numero de clusters), devolvemos
# para cada valor de cluster, el valor de silhouette obtenido
def obtain_silhouette_kmeans(X, cluster_range):
    l = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

        # En labels tenemos para cada punto en de la muestra, a que
        # cluster pertenece
        labels = kmeans.labels_
        l.append(metrics.silhouette_score(X, labels))
    return l


# Dado un rango de valores de epsilon y una métrica, devolvemos para cada
# valor de epsilon el valor de silhouette obtenido
def obtain_silhouette_dbscan(X, eps_range, metric):
    l = []
    clusters = []
    for epsilon in eps_range:
        db = DBSCAN(eps=epsilon, min_samples=10, metric=metric).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Eliminamos el ruido como un posible cluster
        clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
        if clusters[-1] != 1:
            l.append(metrics.silhouette_score(X, labels))
        else:
            l.append(-1)

    return l, clusters


# Ejercicio 1
def exercise_1():

    # Obtenemos la muestra de puntos.
    X, labels_true = obtain_sample()

    # Mostramos los datos en bruto gráficamente.
    cmap = plt.get_cmap('nipy_spectral')
    plt.scatter(X[:, 0], X[:, 1], s=5, marker='o', c=X[:, 0], cmap=cmap)
    plt.title("Sample")
    plt.savefig("3/Sample")
    #plt.show()

    cluster_range = np.arange(1, 16)
    # Para un cluster, el valor de silhouette es = -1, ya que bi = 0
    # por tanto, ese caso lo ponemos ad_hoc
    l = obtain_silhouette_kmeans(X, np.arange(2, 16))
    l.insert(0, -1)

    plt.clf()
    plt.scatter(cluster_range, l, c=cluster_range, cmap=cmap)
    plt.xticks(cluster_range)
    plt.title("Silhouette score per number of clusters with KMeans")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.savefig("3/KMeans (1,16)")
    # plt.show()


# Ejercicio 2
def exercise_2():
    X, labels_true = obtain_sample()
    metrics = ['euclidean', 'manhattan']
    eps_range = np.arange(0.1, 1.0, 0.05)

    # Realizamos el proceso para cada una de las metricas
    for metric in metrics:

        l, clusters = obtain_silhouette_dbscan(X, eps_range, metric)

        # Para cada uno de los valores de (cluster, silhouette)
        # encontrados, se añaden todos los valores de epsilon que tengan
        # asociados dicho par.
        eps_map = {}
        for eps, xy in zip(eps_range, zip(clusters, l)):
            if xy in eps_map:
                eps_map[xy] += ", " + f'{eps:.2f}'
            else:
                eps_map[xy] = f'{eps:.2f}'

        # Generamos para cada punto reconocible, un color.
        norm = matplotlib.colors.Normalize(0, len(eps_map) + 1)
        colors = [plt.get_cmap('nipy_spectral')(norm(i), bytes=True)
                  for i in np.arange(0.5, len(eps_map) + 0.5)]
        colors = [(a / 256, b / 256, c / 256, d / 256) for a, b, c, d in colors]
        col_labels = ['Epsilon values']
        cell_text = [[val] for val in eps_map.values()]
        fig, axs = plt.subplots(2, 1)

        fig.patch.set_visible(False)
        fig.suptitle("Silhouette score per number of clusters with DBSCAN and "
                     + metric + " metric")

        # Situamos arriba la grafica y abajo la tabla
        xy_ = eps_map.keys()
        axs[0].scatter([x for x, _ in xy_], [y for _, y in xy_],
                       s=25, marker='o', c=colors)
        axs[0].set_xlabel("Number of clusters")
        axs[0].set_ylabel("Silhouette score")
        axs[1].axis('off')
        axs[1].axis('tight')
        axs[1].table(cellText=cell_text, colLabels=col_labels,
                                 rowColours=colors, loc='center')

        plt.savefig('3/DBSCAN with '+metric+' metric')
        # plt.show()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
