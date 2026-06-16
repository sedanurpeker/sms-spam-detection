# SMS Spam Detection — Machine Learning and Deep Learning

This repository contains the source code that produces all results, tables, and figures presented in the academic paper **"Detection of SMS Spam Messages Using Machine Learning and Deep Learning Methods"**. Four classical machine learning models and two deep learning models are comparatively evaluated on the UCI SMS Spam Collection dataset.

> **Purpose:** Not to generate new results, but to provide reproducible source code that reliably re-produces the **canonical results** reported in the paper. Seeds and hyperparameters are fixed in `src/config.py`.

---

## 1. Folder Structure

```
sms-spam-detection/
├── README.md                  This file
├── requirements.txt           Python dependencies
├── data/
│   ├── SMSSpamCollection      UCI dataset (5,572 messages)
│   └── UCI_readme.txt         Original dataset description
├── src/
│   ├── config.py              Seeds, paths, hyperparameters
│   ├── data_loader.py         Data loading + EDA statistics
│   ├── preprocessing.py       Text cleaning, TF-IDF, tokenization
│   ├── models.py              6 models + Attention layer
│   ├── evaluate.py            Metrics (Accuracy/Precision/Recall/F1/AUC)
│   ├── figures.py             Code that produces all 7 figures
│   └── main.py                End-to-end execution
└── outputs/
    ├── model_results.csv      Results table (Table 5)
    └── figures/               Figures 1–7 (PNG)
```

## 2. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Running

```bash
cd src
python main.py
```

Upon completion, `outputs/model_results.csv` and the seven PNGs under `outputs/figures/` are re-produced. Total runtime on a single CPU is approximately 1–3 minutes (classical models finish within seconds; deep learning models run for a few epochs with early stopping).

## 4. Dataset

UCI Machine Learning Repository — SMS Spam Collection  
<https://archive.ics.uci.edu/dataset/228/sms+spam+collection>

- Total **5,572** messages: **4,825 ham** (86.6%), **747 spam** (13.4%)
- Format: each line `<label>\t<message>`

## 5. Canonical Results (Table 5)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Naive Bayes | 97.13 | 100.00 | 78.52 | 87.97 | 98.20 |
| Logistic Regression | 97.58 | 96.21 | 85.23 | 90.39 | 98.94 |
| **Linear SVM** | **99.01** | 97.26 | **95.30** | **96.27** | **99.30** |
| Random Forest | 98.21 | 100.00 | 86.58 | 92.81 | 99.11 |
| BiGRU | 98.74 | 97.20 | 93.29 | 95.21 | 98.58 |
| BiGRU + Attention | 97.40 | 92.86 | 87.25 | 89.97 | 97.62 |

## 6. Code ↔ Figure/Table Mapping

| Output | Produced by | Paper section |
|---|---|---|
| Table 5 — `model_results.csv` | `main.py` → `evaluate.py` | Section 7 |
| Figure 1 — Class distribution | `figures.fig_class_distribution()` | Section 3 |
| Figure 2 — Message length | `figures.fig_message_length()` | Section 3 |
| Figure 3 — Word count | `figures.fig_word_count()` | Section 3 |
| Figure 4 — Model comparison | `figures.fig_model_comparison()` | Section 7 |
| Figure 5 — Confusion Matrix | `figures.fig_confusion_matrices()` | Section 7 |
| Figure 6 — ROC curves | `figures.fig_roc_curves()` | Section 7 |
| Figure 7 — Training curves | `figures.fig_training_curves()` | Section 8 |

EDA statistics (Table 1) are documented in `data_loader.dataset_statistics()`, preprocessing steps (Table 3) in `preprocessing.clean_text()`, and hyperparameters (Table 4) in `config.py`.

## 7. Models

**Classical (TF-IDF + scikit-learn):** Naive Bayes (MultinomialNB), Logistic Regression, Linear SVM (LinearSVC + CalibratedClassifierCV for probability/AUC), Random Forest.

**Deep learning (Keras):** BiGRU; BiGRU + Attention (additive/Bahdanau-style attention layer).

## 8. Reproducibility Note

**Classical models** (Naive Bayes, Logistic Regression, Linear SVM, Random Forest with `random_state=42`) are bit-for-bit deterministic: every run of `main.py`, in any environment, will reproduce the values in Table 5 **exactly**.

**Deep learning models** (BiGRU, BiGRU + Attention) are seeded with `tf.random.set_seed(42)`; however, because TensorFlow's multi-threaded CPU operations are not fully deterministic, the metrics for these two models may vary by **up to ~1 F1 point** across runs and environments (e.g. BiGRU F1 ≈ 95.2; BiGRU + Attention F1 ≈ 90). This is a known and expected characteristic of deep learning experiments. What matters is that the **qualitative conclusion remains unchanged**: in every run, Linear SVM is the best-performing model, and the attention mechanism underperforms standard BiGRU on short SMS texts.

The canonical values reported in the paper are fixed in `outputs/model_results.csv` and the accompanying figures submitted with it. To reproduce the deep learning results as closely as possible, it is recommended to use the same TensorFlow version (2.21.0) and a CPU-only environment.