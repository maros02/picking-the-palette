import numpy as np
import anndata
from pathlib import Path
from openTSNE import initialization
from sklearn.datasets import fetch_openml


DATA_PATH = Path(__file__).resolve().parents[1] / "data"

def load_dataset(name, seed):
    name = name.lower()
    if name == "c_elegans":
        return load_h5ad("packer2019.h5ad", "cell_type")
    if name == "mnist":
        return load_mnist()
    if name == "pbmc":
        return load_h5ad("pbmc68k.h5ad", "celltype")
    if name == "tissue":
        return load_h5ad("tissue.h5ad", "cell_type")
    if name == "mouse_brain":
        return load_h5ad("mouse_brain.h5ad", "cluster")

    raise ValueError(f"No such dataset: {name}")


def load_mnist():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32)
    y = y.astype(int)
    return X, y

def load_h5ad(filename, label_col):
    adata = anndata.read_h5ad(DATA_PATH / filename)
    X = adata.X
    labels_str = np.array(adata.obs[label_col])
    _, y = np.unique(labels_str, return_inverse=True)
    return X, y


def pca_reduce(X, n_components: int, random_state: int) -> np.ndarray:
    n_components = min(n_components, X.shape[0], X.shape[1])
    return np.asarray(initialization.pca(X, n_components=n_components, random_state=random_state))