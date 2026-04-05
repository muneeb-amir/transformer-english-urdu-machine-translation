# English → Urdu Machine Translation using Transformers

This repository implements English to Urdu machine translation using two approaches: a custom Transformer model built from scratch and a fine-tuned multilingual pretrained model (mBART-50). The project focuses on comparative evaluation, training dynamics, and efficiency trade-offs.

---

## Overview

The project explores sequence-to-sequence learning using Transformer architectures for neural machine translation. Two distinct approaches are implemented and evaluated:

* Custom Transformer (trained from scratch)
* Fine-tuned mBART-50 (pretrained multilingual model)

The models are compared using BLEU scores and analyzed across corpus-level and sentence-level performance.

---

## Dataset

* English–Urdu parallel corpus (~24,000 sentence pairs)

### Dataset Splits

| Split      | Samples |
| ---------- | ------- |
| Train      | 19,300  |
| Validation | 2,412   |
| Test       | 2,413   |

---

## Model Implementations

### 1. Custom Transformer

* Encoder–Decoder architecture
* Multi-head attention (8 heads)
* Embedding dimension: 256
* Feedforward size: 1024
* Layers: 3 encoder + 3 decoder
* Dropout: 0.1
* Label smoothing: 0.1
* Learning rate scheduler: Noam (warmup = 4000)
* Training: 10 epochs, batch size 32
* Trainable parameters: ~7.8M

---

### 2. mBART-50 Fine-Tuning

* Pretrained multilingual sequence-to-sequence model
* Encoder frozen
* Last ~4 decoder layers fine-tuned
* Optimizer: AdamW (8-bit)
* Learning rate: 5e-5
* Training: 3 epochs, batch size 8
* Trainable parameters: ~324M

---

## Training Setup

* Framework: PyTorch / Hugging Face Transformers
* Mixed Precision Training (AMP)
* GPU acceleration

---

## Evaluation Metrics

* BLEU (Bilingual Evaluation Understudy)
* Sentence-level BLEU statistics

---

## Results

### BLEU Score Comparison (Test Set)

| Metric               | Custom Transformer | mBART-50 |
| -------------------- | ------------------ | -------- |
| Corpus BLEU          | 33.74              | 34.51    |
| Mean Sentence BLEU   | 36.42              | 29.25    |
| Median Sentence BLEU | 18.80              | 12.90    |

---

## Analysis

* mBART-50 achieves slightly higher corpus-level BLEU (+0.77)
* Custom Transformer performs better at sentence-level consistency
* Custom model is significantly more efficient and lightweight
* mBART converges faster due to pretrained knowledge

---

## Key Insights

* Pretrained multilingual models improve robustness and generalization
* Smaller models can outperform at fine-grained sentence-level evaluation
* Trade-off between performance and computational cost is evident

---

## Usage

### Run Training

If using notebook:

* Open `train.ipynb`
* Execute all cells sequentially

---

## Setup

```bash
git clone <repo-url>
cd transformer-english-urdu-machine-translation
pip install -r requirements.txt
```

---

## Highlights

* Dual approach: from-scratch + pretrained model
* Strong comparative analysis with quantitative metrics
* Efficient implementation with clear training dynamics

## Results
FINAL BLEU for both models
<img width="713" height="420" alt="image" src="https://github.com/user-attachments/assets/f7c41021-00b6-4002-9415-789983fb33ae" />

Sentence-level BLEU distribution comparison
<img width="722" height="252" alt="image" src="https://github.com/user-attachments/assets/4cb3eb8e-91b2-4dab-96d5-a00c30989dc8" />


