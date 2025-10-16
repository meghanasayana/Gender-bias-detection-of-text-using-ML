# 03_fairness.py
import numpy as np, datasets, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

# ---------- 1. run inference on EDOS ---------- #
tok   = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").eval()

edos  = datasets.load_from_disk("data/edos")["test"]
enc   = tok(list(edos["sentence"]), padding=True, truncation=True,
            return_tensors="pt")
with torch.no_grad():
    preds = model(**enc).logits.argmax(-1).numpy()
y     = np.array(edos["label"])

# ---------- 2. derive 'protected' column by swapping ---------------- #
def is_female(txt):
    return any(w in txt.lower() for w in ["she","her","woman","girl"])
prot = np.array([1 if is_female(t) else 0 for t in edos["sentence"]])
# 1 = female (unprivileged), 0 = male/default (privileged)

# ---------- 3. wrap into AIF360 ---------- #
bd = BinaryLabelDataset(df=None,
                        favorable_label=0,
                        unfavorable_label=1,
                        label_names=["label"],
                        protected_attribute_names=["prot"])

bd.labels = y.reshape(-1,1)
bd.protected_attributes = prot.reshape(-1,1)
bd.scores = preds.reshape(-1,1)

metric = ClassificationMetric(bd, bd,
                              privileged_groups=[{'prot':0}],
                              unprivileged_groups=[{'prot':1}],
                              predictions=bd.scores)

print({ "SPD": metric.statistical_parity_difference(),
        "EOD": metric.equal_opportunity_difference(),
        "AOD": metric.average_odds_difference(),
        "DI" : metric.disparate_impact() })
