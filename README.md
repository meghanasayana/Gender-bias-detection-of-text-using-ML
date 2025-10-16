# Gender Bias Detection in Text Using Machine Learning

Detect and analyze gender bias in written language using NLP and deep learning. This project leverages transformer models and fairness audits for robust bias identification and mitigation.

***

## 📝 Overview

This project uses modern natural language processing to classify gender bias in text and evaluates fairness using community datasets and audit scripts.
- **Models Used:** RoBERTa transformer
- **Datasets:** Davidson Hate Speech, CrowS-Pairs
- **Key Steps:** Training, bias auditing, fairness analysis, and bias mitigation

***

## 📂 Repository Structure

```
├── 00_download.py       # Download & preprocess datasets
├── 01_train.py          # Train ML model for bias detection
├── 02_bias_audit.py     # Audit bias in predictions
├── 03_fairness.py       # Calculate fairness metrics
├── 04_mitigate.py       # Mitigate bias (Equalized Odds)
├── README.md
├── requirements.txt     # Python dependencies
├── model/               # Saved model/checkpoints
```

***

## 🚀 Getting Started

**Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/gender-bias-detection-ml.git
cd gender-bias-detection-ml
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Data Preparation**
```bash
python 00_download.py
```

**Step 4: Train the Model**
```bash
python 01_train.py
```

**Step 5: Bias Audit**
```bash
python 02_bias_audit.py
```

**Step 6: Fairness Analysis**
```bash
python 03_fairness.py
```

**Step 7: Bias Mitigation**
```bash
python 04_mitigate.py
```

***

## 🔬 Example Usage

Test your model's API with this Python example:
```python
import requests
response = requests.post("https://your-api-url/predict", json={"text":"Nurses are always women and doctors are always men."})
print(response.json())
```

**Sample texts to try:**
- "The nurse comforted the child." (neutral)
- "The scientist presented her findings." (unbiased)
- "He is a strong leader, but she is too emotional." (biased)

***

## 📊 Key Results

- **Model Accuracy:** ~0.95
- **Bias Score (CrowS-Pairs):** 0.435 (anti-stereotype preference)
- **Statistical Parity Difference:** ~-0.02
- **Equal Opportunity Difference:** ~-0.09
- **Mitigation:** Reduced bias metrics with balanced accuracy

***

## 🤝 Contributors

- [meghanasayana](https://github.com/meghanasayana)
- Based on original datasets and open NLP tools cited below

***

## 📚 References

- [CrowS-Pairs Dataset](https://github.com/nyu-mll/crows-pairs)
- [RoBERTa Model](https://huggingface.co/roberta-base)
- [Davidson Hate Speech Dataset](https://github.com/meghanasayana/Gender-bias-detection-of-text-using-ML)

***

## ⚙️ License

MIT License

***

Replace placeholder links/usernames as needed.  
This structure includes setup, usage, examples, technical outcomes, contributor credits, and references for clarity, professionalism, and good open-source practice.
