[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Characterizing Human Semantic Navigation as Trajectories in Embedding Space

Abstract: Semantic representations can be framed as a structured, dynamic knowledge space through which humans navigate to retrieve and manipulate meaning. To investigate how humans traverse this geometry, we introduce a framework that represents concept production as navigation through embedding space. Using different transformer text embedding models, we construct participant-specific semantic trajectories and extract geometric and dynamical metrics—including trajectory, distance to centroid, entropy, velocity, and acceleration. These measures capture both scalar and directional aspects of semantic navigation, providing a computationally grounded view of semantic representation search as movement in a geometric space. We evaluate the framework on four datasets across different languages, spanning different property generation tasks: (i) Neurodegenerative, (ii) Swear verbal fluency, (iii) Property listing task in Italian, and (iv) in German. Across these contexts, our approach distinguishes between clinical groups and concept types, offering a mathematical framework that requires minimal human intervention compared to typical labor-intensive linguistic pre-processing methods. Critically, different embedding models were essentially similar in describing these differences, highlighting similarities between different learned representations despite different training pipelines. By framing semantic navigation as a structured trajectory through embedding space, bridging cognitive modeling with learned representation, thereby establishing a pipeline for quantifying semantic representation dynamics with applications in clinical research, cross-linguistic analysis, and the assessment of artificial cognition.

# Getting Started

Follow these steps to prepare your data, set up the environment, and configure the project for use.

## 1. Add Your Data

Place your files in the `data/raw` folder. If your data is in `.xlsx` format, run the `tocsv.py` script to convert them to `.csv`.

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

Create a `.env` file in the project root with:

```
API_KEY=your-api-key
FILENAME=your-filename
````
