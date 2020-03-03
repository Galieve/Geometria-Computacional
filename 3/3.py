# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:14:02 2020

@author: usu321
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


def obtain_sample():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                                random_state=0)

    return X, labels_true


def obtain_silhouette_kmeans(X, cluster_range):
    l = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
        l.append(metrics.silhouette_score(X, labels))
    return l


def obtain_silhouette_dbscan(X, eps_range, metric):
    l = []
    clusters = []
    for epsilon in eps_range:
        db = DBSCAN(eps=epsilon, min_samples=10, metric=metric).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
        if clusters[-1] != 1:
            l.append(metrics.silhouette_score(X, labels))
        else:
            l.append(-1/len(X))

    return l, clusters


def exercise_1():
    X, labels_true = obtain_sample()
    cluster_range = np.arange(1, 16)

    # For one cluster, silhouette score = -1, as bi = 0
    l = obtain_silhouette_kmeans(X, np.arange(2, 16))
    l.insert(0, -1/len(X))
    plt.scatter(cluster_range, l)
    plt.xticks(cluster_range)
    plt.title("Silhouette score per number of clusters with KMeans")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()


def exercise_2():
    X, labels_true = obtain_sample()
    metrics = ['euclidean', 'manhattan']
    eps_range = np.arange(0.1, 1.0, 0.05)
    for metric in metrics:
        plt.clf()
        l, clusters = obtain_silhouette_dbscan(X, eps_range, metric)
        ax = plt.gca()
        ax.set_xticks(np.arange(1, max(clusters) + 1))
        plt.scatter(clusters, l)
        plt.title("Silhouette score per number of clusters with DBSCAN and " + metric + " metric")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")
        eps_map = {}
        for eps, xy in zip(eps_range,  zip(clusters, l)):
            if xy in eps_map:
                eps_map[xy] += ", "+f'{eps:.2f}'
            else:
                eps_map[xy] = f'{eps:.2f}'
        for xy in eps_map:
            ax.annotate(eps_map[xy], xy=xy, textcoords='data')

        plt.show()



if __name__ == "__main__":
    exercise_1()
    exercise_2()