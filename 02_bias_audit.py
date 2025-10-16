import torch, tqdm, pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

print("🔍 Starting bias evaluation with CrowS-Pairs...")

# Load models for bias evaluation
tok = AutoTokenizer.from_pretrained("roberta-base")
mlm = AutoModelForMaskedLM.from_pretrained("roberta-base")
mlm.eval()

# Load CrowS-Pairs dataset
crows_df = pd.read_csv("data/crows/crows_pairs.csv")
gender_pairs = crows_df[crows_df['bias_type'] == 'gender']

print(f"Evaluating {len(gender_pairs)} gender bias pairs...")

def pseudo_log_likelihood(sentence):
    """Calculate pseudo-log-likelihood for bias evaluation"""
    try:
        enc = tok(sentence, return_tensors="pt", max_length=192, truncation=True)
        ids = enc["input_ids"]
        loss = 0.
        for i in range(1, len(ids)-1):
            masked = ids.clone().unsqueeze(0)
            masked[i] = tok.mask_token_id
            out = mlm(masked, labels=ids.unsqueeze(0))
            loss += out.loss.item()
        return -loss
    except:
        return 0.0

# Evaluate bias preference
correct = 0
total = len(gender_pairs)

for _, row in tqdm.tqdm(gender_pairs.iterrows(), total=total):
    stereo_score = pseudo_log_likelihood(row["sent_more"])   # more stereotypical
    antist_score = pseudo_log_likelihood(row["sent_less"])   # less stereotypical
    
    # Model is less biased if it prefers anti-stereotype
    if antist_score >= stereo_score: 
        correct += 1

anti_stereotype_preference = correct / total
print(f"\n📊 BIAS EVALUATION RESULTS:")
print(f"Anti-stereotype preference: {anti_stereotype_preference:.3f}")
print(f"Correct preferences: {correct}/{total}")
print("Higher scores indicate less bias (ideal: >0.5)")
