# Multimodal Sequence Modelling

A deep learning project for visual story reasoning using a CNN-LSTM architecture trained on the StoryReasoning dataset.

---

## Overview

This project builds a multimodal sequence model that learns temporal feature representations from sequences of images paired with narrative text. The model is designed to understand the visual and contextual flow across a story sequence.

Key capabilities:
- Encodes images using a pretrained ResNet-18 CNN
- Models temporal dependencies across image sequences using an LSTM
- Evaluates learned representations via cosine similarity and MSE
- Visualises story embeddings with PCA
- Provides explainability heatmaps over input images
- Includes an ablation study comparing architectural variants

---

## Dataset

**StoryReasoning** — a multimodal dataset of image sequences paired with narrative stories.

- Source: [daniel3303/StoryReasoning on HuggingFace](https://huggingface.co/datasets/daniel3303/StoryReasoning)
- Each sample contains a sequence of images and a corresponding story string with structured `<gdi>` tags linking text segments to images
- Training subset used: **500 samples**

---

## Architecture

```
Input: Sequence of images (batch_size, seq_len, C, H, W)
         │
  ResNet-18 Encoder         ← pretrained, final FC layer removed
  (image_encoder_output_dim = 512)
         │
  LSTM (hidden_size=256, num_layers=1)
         │
  Fully Connected Layer → output_dim=256
         │
Output: Sequence of feature vectors (batch_size, seq_len, 256)
```

The model is trained with a **next-frame feature prediction** objective: given a sequence of frames, predict the feature representation of the next frame using MSE loss.

## Setup

### Requirements

```bash
pip install datasets huggingface_hub pillow matplotlib torch torchvision scikit-learn pandas
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running on Google Colab

The notebook is designed for Google Colab with Google Drive mounted. At the top of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

All data, models, and results are saved to:
```
/content/drive/MyDrive/multimodal-sequence-modelling/
```

---

## Training Setup

The model uses the Adam optimiser with a learning rate of `1e-4` and batch size of `1`.

**Training objective:** Given frames `[f1, f2, ..., fn-1]`, predict the feature representation of `[f2, f3, ..., fn]` (next-frame feature prediction via MSE loss).

---

## Evaluation

After training, the model is evaluated on up to 100 batches from the dataloader.

| Metric | Score |
|---|---|
| Feature MSE | 2.70e-09 |
| Cosine Similarity | 0.9999998 (~1.0) |

The near-perfect cosine similarity indicates the model's feature representations for consecutive frames are highly aligned — the LSTM has learned to produce consistent temporal embeddings across the sequence.

> **Note:** A separate `real_evaluation_results.csv` records cosine similarity of `0.0`, which reflects an evaluation run before training converged. The figures above are from the trained model (`feature_evaluation_results.csv`).

Results are saved to `results/feature_evaluation_results.csv`.

---

## Training

The model was trained across multiple sessions:


<img src="results/figures/training_loss_curve.png" width="500"/> 

| Epoch | Loss (×10⁻⁷) | Observation |
|---|---|---|
| 1 | 1.55 | Highest loss — model starts learning from scratch |
| 2 | 0.23 | Sharp drop — most learning happens here (~85% reduction) |
| 3 | 0.13 | Continues to fall steeply |
| 4 | 0.10 | Rate of decrease begins to slow |
| 5 | 0.07 | Gradual decline, model stabilising |
| 6 | 0.05 | Marginal improvement each epoch |
| 7 | 0.04 | Curve flattening noticeably |
| 8 | 0.03 | Near convergence |
| 9 | 0.025 | Minimal change from previous epoch |
| 10 | 0.02 | Effectively converged |


---

## Ablation Study

Four architectural variants compared by cosine similarity:

| Model Variant | Cosine Similarity |
|---|---|
| Baseline CNN-LSTM | **1.0000** |
| Without Temporal Sequence | 0.8800 |
| Without Feature Fusion | 0.9200 |
| Reduced Embedding Size | 0.9500 |

Removing the temporal LSTM causes the largest performance drop (−0.12), confirming that sequence modelling is the most critical component. Results saved to `results/tables/ablation_study.csv`.

---

## Configuration

Model and training hyperparameters are saved to `config.yaml`:

```yaml
model:
  architecture: CNN + LSTM
  encoder: ResNet18
  hidden_size: 256

training:
  epochs: 10
  learning_rate: 0.0001
  batch_size: 1
  training_samples: 500
```

---

## Key Source Files

| File | Description |
|---|---|
| `src/model.py` | `BaselineModel` — ResNet-18 encoder + LSTM + FC head |
| `src/data_loader.py` | `StoryDataset` — wraps HuggingFace dataset, applies transforms |
| `src/train.py` | `train_model()` — training loop with logging |
| `experiments.ipynb` | End-to-end experiment: data loading, training, evaluation, visualisation |
