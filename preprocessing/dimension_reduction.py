import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_kmeans_model(dense_model: str, vocab_size: int) -> MiniBatchKMeans:
    """
    Load the KMeans model for a given dense model and vocabulary size.
    If the model does not exist locally, download it from the remote URL.
    """
    os.makedirs(f"assets/kmeans/", exist_ok=True)
    file_name = f"assets/kmeans/km_{dense_model}_{vocab_size}.bin"
    if not os.path.exists(file_name):
        os.system(f"wget https://dl.fbaipublicfiles.com/textless_nlp/gslm/{dense_model}/km{vocab_size}/km.bin")
        os.rename("km.bin", file_name)
    km = joblib.load(file_name)
    return km


def config_dimension_reduction(dense_name: str = "hubert", vocab_sizes: Tuple[int] = (50, 100, 200)) -> None:
    """
    Configures dimension reduction models for a given dense model and vocabulary sizes.
    Generates and saves cluster relationship data and 2D and 3D dimension reduction data to a CSV file.
    """
    centers = []
    km_models = {}
    src_vocab = []
    src_unit = []
    for vocab_size in vocab_sizes:
        km = get_kmeans_model(dense_name, vocab_size)
        centers.append(km.cluster_centers_)
        km_models[vocab_size] = km
        src_vocab.extend([vocab_size] * vocab_size)
        src_unit.extend(list(range(vocab_size)))
    print("Read all Kmeans models")
    centers = np.concatenate(centers)

    clusters_relationship = pd.DataFrame()
    clusters_relationship["src_vocab"] = src_vocab
    clusters_relationship["src_unit"] = src_unit

    for vocab_size in vocab_sizes:
        km = km_models[vocab_size]
        clusters_relationship[f"target_units_{vocab_size}"] = list(
            np.argmin(km.transform(centers), axis=1)
        )
    os.makedirs("assets/dimension_reduction", exist_ok=True)
    clusters_relationship.to_csv(f"assets/dimension_reduction/{dense_name}_clusters.csv")

    for dimension in [2, 3]:
        print(f"Start with {dimension}D dimension reduction")
        dimension_reduction_models = {
            "PCA": PCA(n_components=dimension),
            "T-SNE": TSNE(n_components=dimension, random_state=0),
        }

        for model_name, model in dimension_reduction_models.items():
            print("Starts with ", model_name)
            centers_2d = model.fit_transform(centers)
            centers_2d = pd.DataFrame(
                centers_2d, columns=["X", "Y"] if dimension == 2 else ["X", "Y", "Z"]
            )
            centers_2d["src_vocab"] = src_vocab
            centers_2d["src_unit"] = src_unit
            centers_2d.to_csv(f"assets/dimension_reduction/{dense_name}_{model_name}_{dimension}D.csv")
