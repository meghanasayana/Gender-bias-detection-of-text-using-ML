import numpy as np, datasets, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

print("⚖️ Starting fairness analysis...")

# Load trained model
tok = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model").eval()

if torch.cuda.is_available():
    model = model.cuda()

# Load test data
edos = datasets.load_from_disk("data/edos")["test"]
texts = list(edos["sentence"])
y_true = np.array(edos["label"])

print(f"Analyzing {len(texts)} test samples...")

# Batch inference to avoid memory issues
all_preds = []
batch_size = 32

for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
    batch_texts = texts[i:i+batch_size]
    enc = tok(batch_texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
    
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    
    with torch.no_grad():
        logits = model(**enc).logits
        preds = logits.argmax(-1).cpu().numpy()
        all_preds.extend(preds)

preds = np.array(all_preds)

# Assign protected attributes based on gendered language
def detect_gender_coding(text):
    female_terms = ["she", "her", "woman", "girl", "female", "wife", "mother", "daughter"]
    return any(term in text.lower() for term in female_terms)

protected = np.array([1 if detect_gender_coding(text) else 0 for text in texts])

print(f"Female-coded examples: {protected.sum()}/{len(protected)} ({100*protected.mean():.1f}%)")

# Calculate fairness metrics
def compute_fairness_metrics(y_true, y_pred, protected):
    mask_male = protected == 0
    mask_female = protected == 1
    
    if mask_male.sum() == 0 or mask_female.sum() == 0:
        return {"error": "One group is empty"}
    
    # Statistical Parity Difference
    ppr_male = y_pred[mask_male].mean()
    ppr_female = y_pred[mask_female].mean()
    spd = ppr_female - ppr_male
    
    # Equal Opportunity Difference
    pos_male = y_true[mask_male] == 1
    pos_female = y_true[mask_female] == 1
    
    if pos_male.sum() > 0 and pos_female.sum() > 0:
        tpr_male = y_pred[mask_male][pos_male].mean()
        tpr_female = y_pred[mask_female][pos_female].mean()
        eod = tpr_female - tpr_male
    else:
        eod = float('nan')
    
    # Accuracy by group
    acc_male = (y_pred[mask_male] == y_true[mask_male]).mean()
    acc_female = (y_pred[mask_female] == y_true[mask_female]).mean()
    
    return {
        "Statistical_Parity_Difference": float(spd),
        "Equal_Opportunity_Difference": float(eod),
        "Accuracy_Male": float(acc_male),
        "Accuracy_Female": float(acc_female),
        "Sample_Size_Male": int(mask_male.sum()),
        "Sample_Size_Female": int(mask_female.sum())
    }

metrics = compute_fairness_metrics(y_true, preds, protected)

print("\n📊 FAIRNESS ANALYSIS RESULTS:")
print("=" * 50)
for key, value in metrics.items():
    if isinstance(value, float) and not np.isnan(value):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print(f"\nOverall Accuracy: {(preds == y_true).mean():.4f}")
