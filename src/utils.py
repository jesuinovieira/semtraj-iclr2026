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
        return torch.tensor([], dtype=torch.float32)

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
            print(f"Column '{col}' not found, skipping conversion to torch.Tensor")
            continue

        if not isinstance(df[col].iloc[0], str):
            raise TypeError(f"Column '{col}' must contain strings: {df[col].dtype}")

        df[col] = df[col].apply(str2torch)  # type: ignore

    return df
