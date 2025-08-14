import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import iqr
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances


def _cosine_to_centroid(vectors: torch.Tensor) -> torch.Tensor:
    """Compute the cosine distance of each vector to the centroid of the group.

    Args:
        vectors: unnormalised embeddings of shape (n, dim).

    Returns:
        Tensor of shape (n,) with cosine distance (1‑cos‑sim) to their centroid.
    """
    # Normalise each vector once
    v_norm = F.normalize(vectors, p=2, dim=1)

    # Centroid of those unit vectors
    c = v_norm.mean(dim=0, keepdim=True)
    c_norm = F.normalize(c, p=2, dim=1)  # Keep it on the unit sphere

    # 1 − cosine similarity
    result: torch.Tensor = 1.0 - (v_norm * c_norm).sum(dim=1)  # shape (n,)
    return result


def compute_distances(
    group: pd.DataFrame,
    *,
    add_mds: bool = True,
    remove_outliers: bool = False,
    outlier_threshold: float = 1.5,
    min_points: int = 2,
) -> pd.DataFrame:
    """For each subgroup, compute:

    - Distance to next embedding (cosine)
    - First & second derivatives of that distance
    - Entropy of median‑binarized distances (normalised by series length)
    - Cosine distance to group centroid
    - Optional 2‑D MDS projection (with optional outlier removal)

    # TODO: enhance name or split into multiple functions
    """
    embeddings = torch.stack(group["embedding"].tolist())
    normed = F.normalize(embeddings, p=2, dim=1)

    # Metrics: distance to next
    sims = (normed[:-1] * normed[1:]).sum(1)
    distances = (1.0 - sims).cpu().numpy()

    group = group.copy()
    group["distance_next"] = pd.NA
    group.loc[group.index[:-1], "distance_next"] = distances

    # Metrics: first and second derivatives
    group["distance_derivative"] = group["distance_next"].diff(-1) * -1
    group["distance_second_derivative"] = group["distance_derivative"].diff(-1) * -1

    # Metrics: entropy of median‑split binary series
    valid = group["distance_next"].dropna()
    if len(valid) > 5:
        median_val = valid.median()
        binary = (valid > median_val).astype(int)
        p1 = binary.mean()
        p0 = 1.0 - p1
        if 0 < p1 < 1:
            entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1)) / len(valid)
        else:
            entropy = 0.0
    else:
        entropy = pd.NA
    group["entropy"] = entropy

    # Centroid‑based metrics
    # ----------------------------------------------------------------------------------

    arr = embeddings.cpu().numpy()

    # Metric: distance to order‑centroid (cosine)
    emb_t = torch.stack(group["embedding"].tolist())  # (n, dim)
    group["distance_centroid_order"] = _cosine_to_centroid(emb_t).cpu().numpy()
    arr = embeddings.cpu().numpy()

    # Static centroid (unique properties)
    # ----------------------------------------------------------------------------------

    # Keep first appearance of every Property inside this (ID, Concept) slice
    idx_first = ~group["property"].duplicated()
    uniq_vecs_t = torch.stack(group.loc[idx_first, "prop_embedding"].tolist())
    d_static = _cosine_to_centroid(uniq_vecs_t).cpu().numpy()
    distance_map = dict(zip(group.loc[idx_first, "property"], d_static))
    group["distance_centroid_static"] = group["property"].map(distance_map)

    # Metric: 2‑D MDS
    n = len(group)
    group[["MDS1", "MDS2"]] = np.nan  # initialise

    mask = np.ones(n, dtype=bool)
    if remove_outliers:
        thr = np.median(group["distance_centroid_order"]) + outlier_threshold * iqr(
            group["distance_centroid_order"]
        )
        mask = group["distance_centroid_order"] <= thr
        if mask.sum() < min_points:
            add_mds = False

    if add_mds and mask.sum() >= 2:
        dist_matrix = cosine_distances(arr[mask])

        # Suppress FutureWarnings from MDS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            coords = MDS(
                n_components=2, dissimilarity="precomputed", random_state=42
            ).fit_transform(dist_matrix)

        group.loc[mask, ["MDS1", "MDS2"]] = coords

    return group


def compute_dynamics(group: pd.DataFrame) -> pd.DataFrame:
    """Adds velocity and acceleration (vector + magnitude) to each (ID, Concept)
    subgroup.

    Velocity[i]     = (E[i+1] - E[i]) / (t[i+1] - t[i])
    Acceleration[i] = (V[i+1] - V[i]) / (t[i+2] - t[i+1])

    - Row i gets Velocity; row i also gets Acceleration computed between i → i+1 → i+2.
    - The last row has no velocity; the last two rows have no acceleration.
    """
    # Pre-allocate new columns
    group = group.copy()
    for col in [
        "vel_vector",
        "vel_magnitude",
        "acc_vector",
        "acc_magnitude",
    ]:
        group[col] = pd.NA

    emb = group["embedding"].tolist()

    # Velocity
    vel_vecs = []  # Keep to reuse when computing acceleration
    for i in range(len(emb) - 1):
        v_vec = emb[i + 1] - emb[i]
        v_mag = v_vec.norm().item()

        group.iat[i, group.columns.get_loc("vel_vector")] = v_vec
        group.iat[i, group.columns.get_loc("vel_magnitude")] = v_mag

        vel_vecs.append((i, v_vec))  # store index and dt for later

    # Acceleration (need velocity at i and i + 1, hence loop to len - 2)
    for k in range(len(vel_vecs) - 1):
        i, v_i = vel_vecs[k]
        j, v_ip1 = vel_vecs[k + 1]  # j == i + 1 by construction

        a_vec = v_ip1 - v_i
        a_mag = a_vec.norm().item()

        group.iat[i, group.columns.get_loc("acc_vector")] = a_vec
        group.iat[i, group.columns.get_loc("acc_magnitude")] = a_mag

    return group
