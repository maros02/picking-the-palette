import json
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, directed_hausdorff
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd


# --- Rare class retention

def identify_rare_classes(y, threshold):
    classes, counts = np.unique(y, return_counts=True)
    frequencies = counts / counts.sum()
    return classes[frequencies < threshold]


def rare_class_coverage(y_sample, rare_classes):
    if len(rare_classes) == 0: return None
    return float(np.isin(rare_classes, np.unique(y_sample)).mean())


def avg_rare_class_recall(y, y_sample, rare_classes):
    if len(rare_classes) == 0: return None
    recall = 0.0
    for c in rare_classes:
        recall += np.sum(y_sample == c) / np.sum(y == c)
    return float(recall / len(rare_classes))


def rare_class_purity(emb: np.ndarray, labels: np.ndarray, rare_classes: np.ndarray, k: int):
    rare_mask = np.isin(labels, rare_classes)
    if not rare_mask.any(): return None # no rare classes
    neighbors = knn_indices(emb, k)
    neighbor_labels = labels[neighbors]
    share = (neighbor_labels == labels[:, None]).mean(axis=1)
    return float(share[rare_mask].mean())


def hausdorff_distance(X: np.ndarray, sample_idx: np.ndarray, random_state) -> float:
    if len(sample_idx) == X.shape[0]: return 0.0
    return float(directed_hausdorff(X, X[sample_idx], random_state)[0])


# --- Structure

def class_coverage(y_full: np.ndarray, y_sample: np.ndarray) -> float:
    full_classes = np.unique(y_full)
    sample_classes = np.unique(y_sample)
    return float(np.isin(full_classes, sample_classes).mean())


# source: Espadoto et al. 2019
def neighborhood_hit(emb, labels, k):
    neighbors = knn_indices(emb, k)
    neighbor_labels = labels[neighbors]
    share = (neighbor_labels == labels[:, None]).mean(axis=1)
    return float(share.mean())


# source (code): Baskaran et al. 2022, section 3.5
def cluster_freq_eval(X_full, sample_idx, num_clusters, random_state):
    km = KMeans(init="k-means++", n_clusters=num_clusters, random_state=random_state)
    orig_preds = km.fit_predict(X_full)
    sample_preds = km.predict(X_full[sample_idx])

    cluster_rows = []
    for i in range(num_clusters):
        cluster_rows.append([num_clusters, (orig_preds == i).sum() / orig_preds.shape[0], (sample_preds == i).sum() / sample_preds.shape[0]])

    dfc = pd.DataFrame(cluster_rows, columns=["num_clusters", "original", "sample"])
    avg_l1 = np.mean(np.abs((dfc['sample'] - dfc['original']).values))

    return float(avg_l1)

# --- Faithfulness

def procrustes_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # note: we use the square root
    _, _, disparity = procrustes(emb1, emb2)
    return float(np.sqrt(disparity))


# source (code): Kobak & Berens, 2019
def knn_preservation(X_high, X_low, knn = 10):
    ind1 = knn_indices(X_high, knn)
    ind2 = knn_indices(X_low, knn)

    intersections = 0.0
    for i in range(X_high.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X_high.shape[0] / knn

    return mnn


# source (code): Kobak & Berens, 2019
def knn_class_overlap(X_high: np.ndarray, X_low: np.ndarray, labels, knn_classes: int = 4) -> float | None:
    cl, cl_inv = np.unique(labels, return_inverse=True)
    C = cl.size
    if C < 2: return None
    knn_classes = min(knn_classes, C - 1)

    mu_high = np.zeros((C, X_high.shape[1]))
    mu_low = np.zeros((C, X_low.shape[1]))
    for c in range(C):
        mu_high[c,:] = np.mean(X_high[cl_inv == c], axis=0)
        mu_low[c, :] = np.mean(X_low[cl_inv == c], axis=0)

    nn_high = knn_indices(mu_high, knn_classes)
    nn_low = knn_indices(mu_low, knn_classes)

    intersections = 0.0
    for i in range(C):
        intersections += len(set(nn_high[i]) & set(nn_low[i]))
    mnn_global = intersections / C / knn_classes
    return mnn_global


# source (code): Kobak & Berens, 2019
def cpd_spearman(X_high: np.ndarray, X_low: np.ndarray, random_state, subset_size: int = 1000):
    rng = np.random.default_rng(random_state)
    subset = rng.choice(X_high.shape[0], size=min(subset_size, X_high.shape[0]), replace=False)
    DX = pdist(X_high[subset,:])
    DY = pdist(X_low[subset,:])
    rho = spearmanr(DX[:,None], DY[:,None]).correlation
    return rho


# --- helpers

def grab_subset(sample_size, rng, subset_size: int) -> np.ndarray:
    indices = rng.choice(sample_size, subset_size, replace=False)
    return indices


def knn_indices(emb: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(emb)
    return nn.kneighbors(return_distance=False)


def save_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="ascii")