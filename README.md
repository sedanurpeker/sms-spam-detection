# SMS Spam Detection using Machine Learning and Deep Learning

## Overview

This project presents a comparative analysis of traditional machine learning and deep learning approaches for SMS spam detection. The objective is to automatically classify SMS messages as either **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) techniques and supervised learning models.

A total of six classification models were implemented and evaluated using the SMS Spam Collection Dataset. The performance of each model was compared using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.

---

## Dataset

The project utilizes the **SMS Spam Collection Dataset**, a publicly available benchmark dataset for SMS spam classification.

### Dataset Statistics

| Class            | Count |
| ---------------- | ----: |
| Ham (Legitimate) | 4,825 |
| Spam             |   747 |
| Total            | 5,572 |

### Source

* SMS Spam Collection Dataset
* Almeida et al. (2011)
* UCI Machine Learning Repository

---

## Project Objectives

* Detect spam SMS messages automatically.
* Compare traditional machine learning algorithms with deep learning architectures.
* Evaluate model performance using multiple classification metrics.
* Identify the most effective approach for SMS spam detection.

---

## Methodology

### Data Preprocessing

The following preprocessing steps were applied to the SMS messages:

* Text cleaning
* Lowercase conversion
* Removal of punctuation and special characters
* Tokenization
* Stopword processing
* TF-IDF feature extraction (for machine learning models)
* Sequence tokenization and padding (for deep learning models)

---

## Implemented Models

### Traditional Machine Learning Models

* Naive Bayes
* Logistic Regression
* Linear Support Vector Machine (SVM)
* Random Forest

### Deep Learning Models

* Bidirectional Gated Recurrent Unit (BiGRU)
* BiGRU with Attention Mechanism

---

## Experimental Results

| Model               | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| ------------------- | -----------: | ------------: | ---------: | -----------: | ----------: |
| Naive Bayes         |        97.13 |        100.00 |      78.52 |        87.97 |       98.20 |
| Logistic Regression |        97.58 |         96.21 |      85.23 |        90.39 |       98.94 |
| Linear SVM          |        99.01 |         97.26 |      95.30 |    **96.27** |   **99.30** |
| Random Forest       |        98.21 |        100.00 |      86.58 |        92.81 |       99.11 |
| BiGRU               |        98.74 |         97.87 |      92.62 |        95.17 |       98.79 |
| BiGRU + Attention   |        97.58 |         92.96 |      88.59 |        90.72 |       97.57 |

---

## Key Findings

* Linear SVM achieved the highest overall performance with an F1-Score of 96.27% and a ROC-AUC of 99.30%.
* BiGRU achieved the best performance among deep learning models with an F1-Score of 95.17%.
* Traditional machine learning models demonstrated highly competitive results compared to deep learning architectures.
* The findings indicate that well-optimized machine learning models remain highly effective for text classification tasks involving moderate-sized datasets.

---

## Technologies Used

### Programming Language

* Python 3.12.6

### Libraries and Frameworks

* Scikit-learn
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* NLTK

### Development Environment

* Jupyter Notebook
* Visual Studio Code

---

## Project Structure

```text
sms-spam-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── models/
│
├── outputs/
│   ├── figures/
│   ├── confusion_matrices/
│   └── results/
│
├── src/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/edanurpeker/sms-spam-detection.git
cd sms-spam-detection
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the project:

```bash
python main.py
```

Or open and execute the provided Jupyter notebooks for model training, evaluation, and visualization.

---

## Evaluation Metrics

The following metrics were used to evaluate model performance:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

These metrics provide a comprehensive assessment of classification effectiveness, particularly for imbalanced datasets such as spam detection tasks.

---

## Research Contribution

This study provides a comparative evaluation of machine learning and deep learning methods for SMS spam classification and demonstrates the effectiveness of both approaches on a widely used benchmark dataset.

The project was developed as part of an academic research study in the field of Machine Learning and Natural Language Processing.

---

## Author

**Sedanur Peker**

Computer Engineering

Sakarya University of Applied Sciences

---
