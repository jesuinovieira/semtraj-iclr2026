import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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


def geometric(group: pd.DataFrame, *, cumulative: bool = True) -> pd.DataFrame:
    f_embeddings = "embedding" if cumulative else "prop_embedding"
    embeddings = torch.stack(group[f_embeddings].tolist())
    normed = F.normalize(embeddings, p=2, dim=1)

    # Metric: distance to next
    sims = (normed[:-1] * normed[1:]).sum(1)
    distances = (1.0 - sims).cpu().numpy()

    group = group.copy()
    group["d_next"] = pd.NA
    group.loc[group.index[:-1], "d_next"] = distances

    # Metric: entropy
    valid = group["d_next"].dropna()
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

    # Metric: distance to centroid
    idx_first = ~group["property"].duplicated()
    uniq_vecs_t = torch.stack(group.loc[idx_first, "prop_embedding"].tolist())
    d_static = _cosine_to_centroid(uniq_vecs_t).cpu().numpy()
    distance_map = dict(zip(group.loc[idx_first, "property"], d_static))
    group["d_centroid"] = group["property"].map(distance_map)

    return group


def kinematic(group: pd.DataFrame, *, cumulative: bool = True) -> pd.DataFrame:
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
    for col in ["vel_vector", "vel", "acc_vector", "acc"]:
        group[col] = pd.NA

    emb = group[f_embeddings].tolist()

    # Metric: velocity
    vel_vecs = []  # Keep to reuse when computing acceleration
    for i in range(len(emb) - 1):
        v_vec = emb[i + 1] - emb[i]
        v_mag = v_vec.norm().item()

        group.iat[i, group.columns.get_loc("vel_vector")] = v_vec
        group.iat[i, group.columns.get_loc("vel")] = v_mag

        vel_vecs.append((i, v_vec))  # store index and dt for later

    # Metric: acceleration (need velocity at i and i + 1, hence loop to len - 2)
    for k in range(len(vel_vecs) - 1):
        i, v_i = vel_vecs[k]
        j, v_ip1 = vel_vecs[k + 1]  # j == i + 1 by construction

        a_vec = v_ip1 - v_i
        a_mag = a_vec.norm().item()

        group.iat[i, group.columns.get_loc("acc_vector")] = a_vec
        group.iat[i, group.columns.get_loc("acc")] = a_mag

    return group
