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


def _cosine_threshold(normed: torch.Tensor) -> float:
    """Linz-style subject-specific cutoff δ_p: mean(|cosine|) over all unordered pairs
    of embeddings in this group.

    normed: (n, dim) L2-normalised embeddings
    """
    n = normed.size(0)
    if n < 2:
        return float("nan")

    # Pairwise cosine similarities
    sims = normed @ normed.T  # (n, n), diag == 1

    # Use upper triangle without the diagonal to avoid self-similarity
    idx = torch.triu_indices(n, n, offset=1)
    vals = sims[idx[0], idx[1]].abs()

    return vals.mean().item()


def _cluster_lengths_and_switches(same_edge: np.ndarray, n_tokens: int):
    """
    same_edge: bool array of length n_tokens-1
        same_edge[i] == True means token i and i+1 are in the same cluster.
    n_tokens: total number of tokens in the sequence.
    Returns:
        cluster_lengths: list of lengths (including singletons)
        switches: number of between-cluster transitions (Troyer-style)
    """
    if n_tokens == 0:
        return [], 0
    if n_tokens == 1:
        return [1], 0

    cluster_lengths = []
    current_len = 1

    for i in range(n_tokens - 1):
        if same_edge[i]:
            current_len += 1
        else:
            cluster_lengths.append(current_len)
            current_len = 1

    cluster_lengths.append(current_len)

    # Switches = number of times we start a new cluster
    switches = int((~same_edge).sum())

    return cluster_lengths, switches


def compute_clusters_chains(group: pd.DataFrame) -> pd.DataFrame:
    """Linz-style clustering & chaining for one (id, concept) group.

    Adds per-token columns:
        - cluster_id_chain
        - cluster_id_cluster
        - switch_chain  (boolean, True where a switch occurs)
        - switch_cluster

    And group-level (same value repeated on all rows of the group):
        - mean_cluster_size_chain
        - num_switches_chain
        - mean_cluster_size_cluster
        - num_switches_cluster
    """
    group = group.copy()

    # ------------------------------------------------------------------
    # Prep
    # ------------------------------------------------------------------
    embeddings = torch.stack(group["embedding"].tolist())  # (n, dim)
    normed = F.normalize(embeddings, p=2, dim=1)
    n = normed.size(0)

    # Edge case: too few tokens
    if n < 2:
        group["cluster_id_chain"] = 1 if n == 1 else pd.NA
        group["cluster_id_cluster"] = 1 if n == 1 else pd.NA
        group["switch_chain"] = False
        group["switch_cluster"] = False
        group["mean_cluster_size_chain"] = 0.0
        group["num_switches_chain"] = 0
        group["mean_cluster_size_cluster"] = 0.0
        group["num_switches_cluster"] = 0
        return group

    # ------------------------------------------------------------------
    # Subject-specific cutoff δ_p
    # ------------------------------------------------------------------
    delta = _cosine_threshold(normed)

    # ------------------------------------------------------------------
    # 1) Chain model (adjacent similarity)
    # ------------------------------------------------------------------
    same_chain = np.empty(n - 1, dtype=bool)
    for i in range(1, n):
        # once normalized, cosine similarity is just a dot product
        sim = float((normed[i - 1] * normed[i]).sum())
        print(f"(chain) {sim=}")
        same_chain[i - 1] = abs(sim) > delta
        print(f"{abs(sim)=}, {delta=}, {same_chain[i - 1]=}")

    # Cluster IDs for chain model
    chain_cluster_id = np.empty(n, dtype=int)
    current_id = 1
    chain_cluster_id[0] = current_id
    for i in range(1, n):
        if same_chain[i - 1]:
            chain_cluster_id[i] = current_id
        else:
            current_id += 1
            chain_cluster_id[i] = current_id

    chain_lengths, chain_switches = _cluster_lengths_and_switches(same_chain, n)
    # Troyer: cluster size counted from the second item
    chain_lengths_ge2 = [L for L in chain_lengths if L >= 2]
    if chain_lengths_ge2:
        mean_chain_size = float(np.mean([L - 1 for L in chain_lengths_ge2]))
    else:
        mean_chain_size = 0.0

    # ------------------------------------------------------------------
    # 2) Cluster model (centroid of current cluster)
    # ------------------------------------------------------------------
    same_cluster = np.empty(n - 1, dtype=bool)
    mu = normed[0].clone()
    current_len = 1

    for i in range(1, n):
        # once normalized, cosine similarity is just a dot product
        sim = float((mu * normed[i]).sum())
        print(f"(cluster) {sim=}")
        same = abs(sim) > delta
        same_cluster[i - 1] = same

        if same:
            current_len += 1
            # update centroid of current cluster
            mu = (mu * (current_len - 1) + normed[i]) / current_len
        else:
            # start new cluster
            current_len = 1
            mu = normed[i].clone()

    cluster_cluster_id = np.empty(n, dtype=int)
    current_id = 1
    cluster_cluster_id[0] = current_id
    for i in range(1, n):
        if same_cluster[i - 1]:
            cluster_cluster_id[i] = current_id
        else:
            current_id += 1
            cluster_cluster_id[i] = current_id

    cluster_lengths, cluster_switches = _cluster_lengths_and_switches(same_cluster, n)
    cluster_lengths_ge2 = [L for L in cluster_lengths if L >= 2]
    if cluster_lengths_ge2:
        mean_cluster_size = float(np.mean([L - 1 for L in cluster_lengths_ge2]))
    else:
        mean_cluster_size = 0.0

    # ------------------------------------------------------------------
    # Write results back into the DataFrame
    # ------------------------------------------------------------------
    group["cluster_id_chain"] = chain_cluster_id
    group["cluster_id_cluster"] = cluster_cluster_id

    # Switch flags: first token can’t be a switch by definition
    group["switch_chain"] = np.r_[False, ~same_chain]
    group["switch_cluster"] = np.r_[False, ~same_cluster]

    group["mean_cluster_size_chain"] = mean_chain_size
    group["num_switches_chain"] = int(chain_switches)
    group["mean_cluster_size_cluster"] = mean_cluster_size
    group["num_switches_cluster"] = int(cluster_switches)

    return group


def compute_distances(
    group: pd.DataFrame,
    *,
    cumulative: bool = True,
    add_mds: bool = False,
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
    f_embeddings = "embedding" if cumulative else "prop_embedding"
    embeddings = torch.stack(group[f_embeddings].tolist())
    normed = F.normalize(embeddings, p=2, dim=1)

    # Metrics: distance to next
    sims = (normed[:-1] * normed[1:]).sum(1)
    distances = (1.0 - sims).cpu().numpy()

    group = group.copy()
    group["distance_next"] = pd.NA
    group.loc[group.index[:-1], "distance_next"] = distances

    # Metrics: entropy
    valid = group["distance_next"].dropna()
    if len(valid) > 3:
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
    emb_t = torch.stack(group[f_embeddings].tolist())  # (n, dim)
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
    group[["MDS1", "MDS2"]] = np.nan

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


def compute_dynamics(group: pd.DataFrame, cumulative: bool = True) -> pd.DataFrame:
    """Adds velocity and acceleration (vector + magnitude) to each (ID, Concept)
    subgroup.

    Velocity[i]     = (E[i+1] - E[i]) / (t[i+1] - t[i])
    Acceleration[i] = (V[i+1] - V[i]) / (t[i+2] - t[i+1])

    - Row i gets Velocity; row i also gets Acceleration computed between i → i+1 → i+2.
    - The last row has no velocity; the last two rows have no acceleration.
    """
    f_embeddings = "embedding" if cumulative else "prop_embedding"

    # Pre-allocate new columns
    group = group.copy()
    for col in [
        "vel_vector",
        "vel_magnitude",
        "acc_vector",
        "acc_magnitude",
    ]:
        group[col] = pd.NA

    emb = group[f_embeddings].tolist()

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
