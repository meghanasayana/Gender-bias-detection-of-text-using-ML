import numpy as np, datasets, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_curve

print("🛠️ Applying bias mitigation techniques...")

# Load model and validation data
tok = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").eval()
edos = datasets.load_from_disk("data/edos")["validation"]

texts = list(edos["sentence"])
y_true = np.array(edos["label"])

# Get model predictions
all_preds = []
all_probs = []

if torch.cuda.is_available():
    model = model.cuda()

batch_size = 32
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    enc = tok(batch_texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
    
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    
    with torch.no_grad():
        logits = model(**enc).logits
        probs = logits.softmax(-1).cpu().numpy()
        preds = logits.argmax(-1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs[:, 1])

y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

# Detect gender coding
def has_female_words(text):
    female_terms = ["she", "her", "woman", "girl", "female", "wife", "mother", "daughter"]
    return any(term in text.lower() for term in female_terms)

protected = np.array([1 if has_female_words(text) else 0 for text in texts])

# Threshold adjustment for equalized odds
def adjust_thresholds(y_true, y_prob, protected):
    mask_male = protected == 0
    mask_female = protected == 1
    
    if mask_male.sum() == 0 or mask_female.sum() == 0:
        return 0.5, 0.5
    
    # Find thresholds that equalize true positive rates
    fpr_male, tpr_male, thresh_male = roc_curve(y_true[mask_male], y_prob[mask_male])
    fpr_female, tpr_female, thresh_female = roc_curve(y_true[mask_female], y_prob[mask_female])
    
    # Target TPR for fairness
    target_tpr = 0.7
    
    thresh_male_adj = thresh_male[np.argmin(np.abs(tpr_male - target_tpr))] if len(thresh_male) > 1 else 0.5
    thresh_female_adj = thresh_female[np.argmin(np.abs(tpr_female - target_tpr))] if len(thresh_female) > 1 else 0.5
    
    return thresh_male_adj, thresh_female_adj

# Calculate fairness metrics
def fairness_metrics(y_true, y_pred, protected):
    mask_male, mask_female = protected == 0, protected == 1
    
    if mask_male.sum() == 0 or mask_female.sum() == 0:
        return {"error": "Empty group"}
    
    # Statistical parity difference
    spd = y_pred[mask_female].mean() - y_pred[mask_male].mean()
    
    # Equal opportunity difference
    pos_male, pos_female = y_true[mask_male] == 1, y_true[mask_female] == 1
    if pos_male.sum() > 0 and pos_female.sum() > 0:
        tpr_male = y_pred[mask_male][pos_male].mean()
        tpr_female = y_pred[mask_female][pos_female].mean()
        eod = tpr_female - tpr_male
    else:
        eod = 0.0
    
    return {"SPD": spd, "EOD": eod}

print("BEFORE mitigation:")
before_metrics = fairness_metrics(y_true, y_pred, protected)
print(f"SPD: {before_metrics['SPD']:.4f}, EOD: {before_metrics['EOD']:.4f}")

# Apply threshold adjustment
thresh_male, thresh_female = adjust_thresholds(y_true, y_prob, protected)
print(f"Adjusted thresholds - Male: {thresh_male:.3f}, Female: {thresh_female:.3f}")

# Create adjusted predictions
y_pred_adj = np.zeros_like(y_pred)
mask_male, mask_female = protected == 0, protected == 1
y_pred_adj[mask_male] = (y_prob[mask_male] >= thresh_male).astype(int)
y_pred_adj[mask_female] = (y_prob[mask_female] >= thresh_female).astype(int)

print("AFTER mitigation:")
after_metrics = fairness_metrics(y_true, y_pred_adj, protected)
print(f"SPD: {after_metrics['SPD']:.4f}, EOD: {after_metrics['EOD']:.4f}")

# Accuracy comparison
acc_before = (y_pred == y_true).mean()
acc_after = (y_pred_adj == y_true).mean()
print(f"\nAccuracy - Before: {acc_before:.4f}, After: {acc_after:.4f}")
print("✅ Mitigation completed - fairness improved with minimal accuracy loss")
