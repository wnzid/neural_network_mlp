achieved strong results, with accuracies ranging from 92.50% to 93.50%. The best performance 
# MLP Classification — dataset_039

## Overview

An MLP (multi-layer perceptron) was trained to classify the samples in `dataset_039.csv`. The
data pipeline included a train/test split and feature normalization. Three experiments were
conducted using different hyperparameters and architectures; all runs produced strong results.

## Key Results

- Accuracy range: **92.50% — 93.50%**
- Best model: **mlp_run_2** (see `mlp_results/`)

## Experiments

- Experiment A — Altered architecture and learning rate
- Experiment B — Regularization and different hidden sizes
- Experiment C — Deeper network variant

All experiments followed the same preprocessing steps; differences were limited to model
architecture and training hyperparameters.

## Files

- `dataset_039.csv` — Input dataset
- `neural_network.py` — Training and evaluation script
- `mlp_results/` — Saved model checkpoints (`mlp_run_1.pth`, `mlp_run_2.pth`, `mlp_run_3.pth`)

## Reproduce

Run training/evaluation locally with:

```bash
python neural_network.py
```

Adjust experiment settings inside `neural_network.py`

## Notes

This repository presents a streamlined MLP workflow — from data preparation and normalization
to experimental evaluation and model persistence — completed as an assignment for the
"Artificial Intelligence" course at Vilnius University.