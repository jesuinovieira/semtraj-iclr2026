import pandas as pd
import torch
from tqdm.auto import tqdm

import backends


def embed(
    backend: backends.EmbeddingBackend, series: pd.Series, batch_size: int = 100
) -> list[torch.Tensor]:
    """Compute embeddings for a Pandas Series in batches.

    The function splits the input Series into batches, sends each batch for embedding
    computation using a provided backend, and converts the resulting embeddings into
    `torch.Tensor` objects (dtype=float32) for downstream PyTorch operations.

    Args:
        backend: The embedding service backend used to fetch embeddings.
        series: A pandas Series of text strings to embed.
        batch_size: Number of items per batch. Defaults to 100.

    Returns:
        A list where each element is a torch.Tensor representing the embedding vector of
        the corresponding input text.
    """
    embeddings: list[torch.Tensor] = []

    for start in tqdm(range(0, len(series), batch_size), desc="Embedding batches"):
        # Extract the current batch of texts
        texts = series.iloc[start : start + batch_size].tolist()

        # Embed the batch using the provided backend
        embeds = backend.embed(texts)

        # Convert each embed list to torch.Tensor (float32)
        embeddings.extend(torch.tensor(e, dtype=torch.float32) for e in embeds)

    return embeddings


def cumulative_concat(properties: pd.Series) -> list[str]:
    """Build a running concatenation of property strings for each time step."""
    accumulated: list[str] = []
    result: list[str] = []
    for prop in properties:
        accumulated.append(prop)
        result.append(" ".join(accumulated))
    return result


def get_lang_from_filename(filename: str) -> str:
    if "Italian" in filename:
        return "it"
    if "German" in filename:
        return "de"
    if "Parkinson" in filename:
        return "es"

    return "en"


def torch2str(t: torch.Tensor) -> str:
    """Convert a 1D torch.Tensor to a comma-separated string.

    Args:
        t: 1D tensor.

    Returns:
        Comma-separated string representation.
    """
    if t is None or pd.isna(t):  # type: ignore
        return ""

    if t.ndim != 1:
        raise ValueError("torch2str only supports 1D tensors.")

    return ",".join(map(str, t.tolist()))


def str2torch(s: str) -> torch.Tensor:
    """Convert a comma-separated string back to a 1D torch.Tensor.

    Args:
        s: Comma-separated string of floats.

    Returns:
        1D tensor with dtype float32.
    """
    if s is None or s == "":
        return torch.tensor([float("nan")], dtype=torch.float32)

    if isinstance(s, float):
        return torch.tensor([s], dtype=torch.float32)

    return torch.tensor([float(v) for v in s.split(",")], dtype=torch.float32)


def save(df: pd.DataFrame, dst: str) -> None:
    """Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        dst: Path to the output CSV file.
    """
    # TODO: torch2str or torch2numpy, what's better?
    for col in df.columns:
        if not isinstance(df[col].iloc[0], torch.Tensor):
            continue
        df[col] = df[col].apply(torch2str)

    df.to_csv(dst, index=False)


def load(
    src: str,
    tensor_cols: list[str] = [
        "embedding",
        "prop_embedding",
        "vel_vector",
        "acc_vector",
    ],
) -> pd.DataFrame:
    """Load a DataFrame from a CSV file.

    Args:
        src: Name of the input CSV file.
        tensor_cols: List of columns to convert back to torch.Tensor.

    Returns:
        DataFrame with embeddings converted back to torch.Tensor.
    """
    df = pd.read_csv(src)

    for col in tensor_cols:
        if col not in df.columns:
            continue

        if not isinstance(df[col].iloc[0], str):
            raise TypeError(f"Column '{col}' must contain strings: {df[col].dtype}")

        df[col] = df[col].apply(str2torch)  # type: ignore

    return df


def zca_whitened_embeddings(
    df: pd.DataFrame,
    emb_col: str = "embedding",
    new_col: str = "embedding_zca",
    device: str | None = None,
    dtype=torch.float32,
) -> tuple[pd.DataFrame, dict]:
    """Fit a ZCA-whitening transform on all embeddings in `emb_col` and add a new column
    `new_col` with the whitened embeddings.

    # TODO: rename and refactor function

    - df[emb_col] is expected to be a Series of 1D torch tensors of shape (d,).
    - Returns a *copy* of df with the new column, plus a dict containing
      the mean and whitening matrix so you can reuse them elsewhere.
    """
    # 0) Copy to avoid mutating the original df in-place
    df_out = df.copy()

    # 1) Stack all embeddings into a single tensor (n_samples, d)
    X_list = df_out[emb_col].tolist()
    X = torch.stack([x.to(dtype=dtype) for x in X_list], dim=0)

    # 2) Device handling
    if device is None:
        device = X.device
    X = X.to(device)

    # 3) Mean and centering
    mean = X.mean(dim=0, keepdim=True)  # (1, d)
    Xc = X - mean  # (n, d)

    # 4) Covariance (d x d)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / (n - 1)

    # 5) Eigen-decomposition (symmetric)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # 6) Sort eigenvalues descending (optional but nice)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eps = eigvals.max() * 1e-5  # 0.00001 × max eigenvalue

    # 7) Build whitening matrix W_ZCA = U Λ^{-1/2} U^T
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + eps))
    W = eigvecs @ inv_sqrt @ eigvecs.T  # (d, d)

    # 8) Apply whitening to all embeddings
    Z = (X - mean) @ W.T  # (n, d)

    # 9) Put back into DataFrame as torch tensors
    df_out[new_col] = [z for z in Z]

    # 10) Return df and the transform parameters (for reuse on test data)
    params = {"mean": mean, "W": W, "device": device, "dtype": dtype}

    # 11) Diagnostics
    cov_z = (Z.T @ Z) / (n - 1)
    print(f"Whitened cov diagonal mean (~1): {cov_z.diag().mean():.4f}")
    print(
        f"Whitened cov off-diagonal mean (~0): {cov_z.fill_diagonal_(0).abs().mean():.4f}"
    )
    print(f"eps used: {eps:.2e}")
    print(f"n samples: {n}, embedding dim: {X.shape[1]}")

    return df_out, params
