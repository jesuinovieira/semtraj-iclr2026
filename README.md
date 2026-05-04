[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space

> Semantic representations can be framed as a structured, dynamic knowledge space through which humans navigate to retrieve and manipulate meaning. To investigate how humans traverse this geometry, we introduce a framework that represents concept production as navigation through embedding space. Using different transformer text embedding models, we construct participant-specific semantic trajectories based on cumulative embeddings and extract geometric and dynamical metrics, including distance to next, distance to centroid, entropy, velocity, and acceleration. These measures capture both scalar and directional aspects of semantic navigation, providing a computationally grounded view of semantic representation search as movement in a geometric space. We evaluate the framework on four datasets across different languages, spanning different property generation tasks: Neurodegenerative, Swear verbal fluency, Property listing task in Italian, and in German. Across these contexts, our approach distinguishes between clinical groups and concept types, offering a mathematical framework that requires minimal human intervention compared to typical labor-intensive linguistic pre-processing methods. Comparison with a non-cumulative approach reveals that cumulative embeddings work best for longer trajectories, whereas shorter ones may provide too little context, favoring the non-cumulative alternative. Critically, different embedding models yielded similar results, highlighting similarities between different learned representations despite different training pipelines. By framing semantic navigation as a structured trajectory through embedding space, bridging cognitive modeling with learned representation, thereby establishing a pipeline for quantifying semantic representation dynamics with applications in clinical research, cross-linguistic analysis, and the assessment of artificial cognition.

_Project page: https://rodrigo-motta.github.io/semtraj-iclr2026-page/_

# Getting Started

Follow these steps to prepare your data, set up the environment, and configure the project for use.

## 1. Add Your Data

Place your files in the `data/raw` folder. If your data is in `.xlsx` format, run the `data/raw/tocsv.py` script to convert them to `.csv`.

> **Note:** The Parkinson dataset was obtained from Toro-Hernández et al., [*Neurocognitive correlates of semantic memory navigation in Parkinson's disease*](https://www.nature.com/articles/s41531-024-00630-4), *npj Parkinson's Disease* (2024).

## 2. Set Up the Environment

```bash
# Install uv (if not already installed)
pip install uv

# Create and activate a virtual environment
uv venv -p 3.11 .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
```

## 3. Configure Environment Variables

Create a `.env` file in the project root:

```
BACKEND=openai              # openai | gemini | qwen | fasttext
API_KEY_OPENAI=sk-...
API_KEY_GEMINI=...
FILENAME=german             # parkinson | swear-fluency | italian | german
CUMULATIVE=True             # cumulative vs. stepwise trajectories
```

Only the API key for the selected `BACKEND` is required. `qwen` and `fasttext` run locally and need no key.

## 4. Run the Pipeline

Notebooks are numbered and meant to run in order:

| Notebook | Purpose |
|---|---|
| [01-eda.ipynb](notebooks/01-eda.ipynb) | Exploratory data analysis |
| [02-embed.ipynb](notebooks/02-embed.ipynb) | Build embeddings for the selected backend/dataset |
| [03-metrics.ipynb](notebooks/03-metrics.ipynb) | Compute trajectory, centroid distance, entropy, velocity, acceleration |
| [04-analysis-*.ipynb](notebooks/) | Boxplots, heatmaps, significance tests |
| [05-3dviz.ipynb](notebooks/05-3dviz.ipynb) | 3D trajectory visualization |

To batch-execute across every `BACKEND × FILENAME × CUMULATIVE` combination, use the [Makefile](Makefile):

```bash
make run        # embed + metrics + analysis
make embed      # just step 02
make metrics    # just step 03
make analysis   # just step 04 (boxplots)
make clean      # wipe notebooks/executed/
```

Executed notebooks land in `notebooks/executed/`.

## Citation

```bibtex
@inproceedings{
    toro-hernandez2026characterizing,
    title={Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space},
    author={Felipe Diego Toro-Hern{\'a}ndez and Jesuino Vieira Filho and Rodrigo M. Cabral-Carvalho},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=QQVmIR97sf}
}
```
