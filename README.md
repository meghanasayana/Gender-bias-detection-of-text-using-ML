# Gender Bias Detection of Text Using Machine Learning

Detect, audit, and mitigate gender bias in textual data using state-of-the-art NLP techniques, transformer models, and fairness analysis.

---

## Overview

This project aims to identify, analyze, and mitigate gender bias in text using machine learning and deep learning models. It leverages the [Davidson Hate Speech Dataset](https://github.com/meghanasayana/Gender-bias-detection-of-text-using-ML), [CrowS-Pairs dataset](https://github.com/nyu-mll/crows-pairs), and transformer models (RoBERTa) for sequence classification.

---

## Folder Layout

```text
Gender-bias-detection-of-text-using-ML/
│
├── 00_download.py       # Download and prepare the datasets
├── 01_train.py          # Train bias detection model (RoBERTa)
├── 02_bias_audit.py     # Audit bias using CrowS-Pairs
├── 03_fairness.py       # Fairness analysis and metrics
├── 04_mitigate.py       # Mitigation script (Equalized Odds)
├── requirements.txt     # Python package requirements
├── README.md
└── model/               # Saved models and tokenizer
```

---

## Setup

### Clone the repository

```bash
git clone https://github.com/meghanasayana/Gender-bias-detection-of-text-using-ML.git
cd Gender-bias-detection-of-text-using-ML
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

- **Hate Speech Detection Dataset**: Preprocessed for binary classification.
- **CrowS-Pairs**: Used for bias auditing. Focus on sentences related to gender.

---

## Running the Pipeline

### Step 1: Download and Prepare Datasets

```bash
python 00_download.py
```

### Step 2: Train the Model

```bash
python 01_train.py
```

### Step 3: Bias Audit

```bash
python 02_bias_audit.py
```

### Step 4: Fairness Analysis

```bash
python 03_fairness.py
```

### Step 5: Bias Mitigation (Equalized Odds)

```bash
python 04_mitigate.py
```

---

## Key Results & Metrics

- **Model test accuracy**: ~0.95
- **CrowS-Pairs (gender bias audit)**: Anti-stereotype preference ≈ 0.435 (ideal >0.5)
- **Fairness Analysis**:
  - Statistical Parity Difference: ≈ -0.02
  - Equal Opportunity Difference: ≈ -0.09
  - Accuracy Female-Coded: ≈ 0.96
  - Accuracy Male-Coded: ≈ 0.94
- **Mitigation**: Threshold adjustment reduces SPD/EOD while balancing accuracy.

---

## Interpretation Guide

- **Statistical Parity Difference (SPD)**: Close to 0 means equal positive rates across groups.
- **Equal Opportunity Difference (EOD)**: Low absolute value = less bias; large values = more bias.
- **Post-mitigation**: Look for reduction in bias metrics with minimal loss of accuracy.

---

## Authors & Credits

- [Meghana Sayana](https://github.com/meghanasayana)
- Based on CrowS-Pairs and [Davidson Hate Speech dataset](https://github.com/meghanasayana/Gender-bias-detection-of-text-using-ML)

**Colab Notebook**: [View on Google Colab](https://colab.research.google.com/drive/1z-IYJ0ysK-UxdTPTAl5D9ZsaKu0I0t8Z#updateTitle=true&folderId=13BYgcSf-8-z3-FybKA5MfmJTAKifyffU&scrollTo=ONpJynWbSGxj)

---

## License

This project is MIT Licensed.

---

## References

- [CrowS-Pairs Dataset](https://github.com/nyu-mll/crows-pairs)
- [RoBERTa Model](https://huggingface.co/roberta-base)
- [Transformers Library](https://github.com/huggingface/transformers)
