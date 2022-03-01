import MySQLdb
import numpy as np
import matplotlib.pyplot as plt

# ===== test vars ====
geno = 0
gene = "AT1G01080.1"
restriction = [0, 1, 2, 3]

search = list(set([i for i in range(0, 7)]) - set(restriction))
# ==== code starts here ====


def get_knn_query(cursor, agi, genotype):
    """
    Get query for knn, i.e mean abundance for a gene + genotype
    :return: numpy array of mean abundances
    """
    q = "SELECT * FROM abundance WHERE AGI LIKE \'%" + agi + "%\'"
    q += " AND genotype=" + str(genotype) + ";"
    cursor.execute(q)

    raw = cursor.fetchall()

    q = np.array(raw)[:, 4].astype(float)
    q = np.reshape(q, (12, 4)).T
    q = np.nanmean(q, axis=0)

    return q


def get_abundance_from_genotype(cursor, genotype):
    """
    Get array of abundances given a genotype
    :return: array of shape (# genes, # timepoints), array of AGIs
    """
    q = "SELECT * FROM abundance WHERE genotype=" + str(genotype) + ";"
    cursor.execute(q)
    raw = cursor.fetchall()
    raw = np.array(raw)

    abundance = raw[:, 4].astype(float)
    abundance = np.reshape(abundance, (12, 4, 4482))
    abundance = abundance.T
    abundance[abundance == 0.0] = np.nan
    abundance = np.nanmean(abundance, axis=1)

    genes = raw[:, 1].reshape(48, 4482).T
    genes = genes[:, 0]

    return abundance, genes


def normalize(arr):
    """
    normalize array with mean and stdev.
    Array has shape of (samples, timepoints)
    """
    mean = np.array([np.nanmean(arr, axis=1)]).T
    stdev = np.array([np.nanstd(arr, axis=1)]).T

    mean = np.tile(mean, (1, 12))
    stdev = np.tile(stdev, (1, 12))

    arr = (arr - mean) / stdev

    return arr


def knn(x, arr, k):
    """
    return k nearest neighbors of array given query x
    Array has shape of (samples, timepoints)
    Return array of indices of k closest series
    """

    dist = (x - arr)
    dist = np.sum(dist*dist, axis=1)

    available = 12 / (12 - np.sum(np.isnan(arr).astype(int), axis=1))
    weighted_dist = available*dist

    h = list(np.argpartition(weighted_dist, k)[:k])

    return np.argpartition(weighted_dist, k)[:k]


# ===== initialize ====
con = MySQLdb.connect('localhost', 'root', '', 'arabidopsis_proteomics')
cur = con.cursor()
query = get_knn_query(cur, gene, geno)
dset = []
gene_lookup = []
genotype_partition = []
res = []

# load relevant genotypes
for g in search:
    curr_abundance, curr_genes = get_abundance_from_genotype(cur, g)
    dset.append(curr_abundance)
    gene_lookup.append(curr_genes)
    if len(genotype_partition) == 0:
        genotype_partition.append(curr_genes.shape[0])
    else:
        genotype_partition.append(genotype_partition[-1] + curr_genes.shape[0])

# run knn on dataset
dset = normalize(np.concatenate(dset, axis=0))
gene_lookup = np.concatenate(gene_lookup, axis=0)
query2 = (query - np.nanmean(query)) / np.nanstd(query)
nearest = knn(query2, dset, 20)

# process result
for n in nearest:
    curr = {"prot_group": gene_lookup[n]}
    curr_g = 0
    for i in range(0, len(search)):
        if n < genotype_partition[i]:
            curr_g = i

    curr["genotype"] = search[curr_g]
    curr["values"] = dset[n]
    res.append(curr)

