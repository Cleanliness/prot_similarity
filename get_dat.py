import pandas
import numpy as np
import torch
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 0 - wt, 1 ...
def get_geno(g, df):
    """
    Given abundance dataframe, convert to np array.
    shape = prot, replicant, zt
    """

    c = 0
    res = []
    zt = []
    first = True
    for i in df.columns[1 + g*48: 1 + (g + 1)*48]:
        curr = df[i]

        if c % 4 == 0 and not first:
            reps = np.array(zt)
            res.append(reps)
            zt = []
        zt.append(np.array(curr))
        c += 1
        first = False

    res.append(np.array(zt))
    cleaned = []
    for r in np.array(res).T:
        m = np.nanmean(r, axis=0)
        if np.count_nonzero(np.isnan(m)) < 6:
            cleaned.append(m)

    return np.array(cleaned)


def normalize(arr):
    """
    Normalize array from get_geno
    """
    mean = np.array([np.nanmean(arr, axis=1)]).T
    stdev = np.array([np.nanstd(arr, axis=1)]).T

    mean = np.tile(mean, (1, 12))
    stdev = np.tile(stdev, (1, 12))

    arr = (arr - mean) / stdev

    return arr


def knn(arr, x, k):
    """
    get k closest neighbours to x
    """

    flattened_x = np.array([x])
    flattened = arr

    dist = nan_euclidean_distances(flattened_x, flattened)

    # remove query point
    sort_indices = np.argsort(dist[0])

    return sort_indices[1: k+1]


def plot_points(arr):
    xpoints = []
    ypoints = []
    for i in range(0, 12):
        x = 2*i + 1

        xpoints.append(x)
        ypoints.append(arr[i])

    plt.scatter(xpoints, ypoints)

f = open("ab.csv")
dataframe = pandas.read_csv(f)
dset = get_geno(0, dataframe)
g = normalize(dset)

