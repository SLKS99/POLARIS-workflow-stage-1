# POLARIS Workflow: LLM Literature Mining → Database → MT-DKL GNN + Sparse GP

This repository contains the full workflow used in our study: (1) LLM-assisted literature extraction and benchmarking, (2) database construction and graph featurization, and (3) multi-task deep kernel learning with a graph neural network encoder and sparse Gaussian process heads for uncertainty-aware screening and active learning.

## Overview

The workflow is organized into three stages:

1) Literature Mining and Extraction Benchmarking  
We extract perovskite spacer and property information from a corpus of papers using multiple LLM tools, then quantify extraction quality using ground-truth labels and standardized scoring.

2) Database Construction  
We build a combined molecular dataset consisting of literature-derived perovskite spacer cations and an augmented set of ammonium-containing molecules from QM9. We standardize identifiers, attach labels and flags, and generate graph objects used by the ML pipeline.

3) Modeling and Screening  
We train a multi-task deep kernel learning model consisting of a shared GNN encoder and task-specific sparse GP heads. We optionally run active learning to propose diverse, high-uncertainty candidates for follow-up.

## Repository Contents

- `extraction/`  
  Scripts and templates for LLM-based extraction, prompt formats, parsing, and post-processing.

- `evaluation/`  
  Scoring and benchmarking utilities (token F1, phrase recall, ranking across models, stability metrics).

- `data/`  
  Input and intermediate CSV files. Typical examples:
  - `perov_extracted.csv` (literature-derived spacers and properties)
  - `qm9_filtered.csv` (QM9 ammonium-containing subset)
  - `combined_dataset.csv` (concatenated dataset with flags)

- `graphs/`  
  Graph tensors / serialized graph objects used for training.
  - `combined_graphs.pt`

- `models/`  
  Saved model checkpoints.
  - `final_dkl_model.pt`

- `notebooks/`  
  End-to-end analysis and training notebooks.
  - `DKL_GNN_multitask_sparseGP.ipynb`

- `figures/`  
  Plot scripts and outputs used for the manuscript and SI.

## Requirements

- Python 3.9+ recommended
- Common packages:
  - numpy, pandas, matplotlib
  - scikit-learn (metrics, optional baselines)
  - torch (PyTorch)
  - torch-geometric (graph featurization and message passing)
  - gpytorch (Gaussian processes)
  - tqdm

If you use a conda environment, a typical setup is:

```bash
conda create -n polaris python=3.10 -y
conda activate polaris
pip install numpy pandas matplotlib scikit-learn tqdm
pip install torch
pip install torch-geometric
pip install gpytorch

For questions or issues, contact:

sheryl Sanchez [sheryl.leakelly@gmail.com]
