# 04_mitigate.py
import numpy as np, datasets, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.metrics import ClassificationMetric

tok   = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").eval()
edos  = datasets.load_from_disk("data/edos")["validation"]

enc   = tok(list(edos["sentence"]), padding=True, truncation=True,
            return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits
y_true = np.array(edos["label"])
y_pred = logits.argmax(-1).numpy()

prot   = np.array([1 if "she" in t.lower() or "her" in t.lower() else 0
                   for t in edos["sentence"]])

d_true = BinaryLabelDataset(df=None, label_names=["label"],
                            protected_attribute_names=["prot"],
                            favorable_label=0, unfavorable_label=1)
d_pred = d_true.copy()
d_true.labels = y_true.reshape(-1,1); d_pred.labels = y_pred.reshape(-1,1)
d_true.protected_attributes = d_pred.protected_attributes = prot.reshape(-1,1)

eq = EqOddsPostprocessing(unprivileged_groups=[{'prot':1}],
                          privileged_groups=[{'prot':0}], seed=42)
eq.fit(d_true, d_pred)
d_adj = eq.predict(d_pred)
y_adj = d_adj.labels.ravel()

m_before = ClassificationMetric(d_true, d_pred,
            privileged_groups=[{'prot':0}], unprivileged_groups=[{'prot':1}])
m_after  = ClassificationMetric(d_true, d_adj,
            privileged_groups=[{'prot':0}], unprivileged_groups=[{'prot':1}])

print("Equalized Odds applied")
print({"SPD_before": m_before.statistical_parity_difference(),
       "SPD_after" : m_after.statistical_parity_difference(),
       "EOD_before": m_before.equal_opportunity_difference(),
       "EOD_after" : m_after.equal_opportunity_difference()})
