# Gender Bias Detection in Text (EDOS + CrowS-Pairs)

## Folder Layout
```text
gender-bias-nlp/
├── requirements.txt
├── 00_download.py
├── 01_train.py
├── 02_bias_audit.py
├── 03_fairness.py
├── 04_mitigate.py
└── README.md
```

## Quick Start

Follow these steps to set up and run the gender bias detection system:

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

```bash
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and cache datasets

```bash
python 00_download.py
```

### 4. Fine-tune the RoBERTa model

```bash
python 01_train.py
```

### 5. Run gender bias audit using CrowS-Pairs

```bash
python 02_bias_audit.py
```

### 6. Calculate fairness metrics on EDOS test set

```bash
python 03_fairness.py
```

### 7. Apply equalized-odds post-processing mitigation

```bash
python 04_mitigate.py
```

## What is Included

* **EDOS Sexism Detection Dataset** – The SemEval-2023 Task 10 dataset for identifying sexism in text[4].
* **CrowS-Pairs Minimal Pairs** – A dataset for auditing gender bias through minimal pair comparisons[6][12].
* **AIF360 Metrics** – Comprehensive fairness metrics from IBM's AI Fairness 360 toolkit[7].
* **Equalized Odds Mitigation** – Post-processing technique to reduce disparate impact across demographic groups[7].

**Note:** Results and plots generated from these scripts can be directly incorporated into your research report or analysis documentation.
