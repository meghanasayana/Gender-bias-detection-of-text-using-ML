# 02_bias_audit.py
import torch, tqdm, datasets
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained("roberta-base")
mlm = AutoModelForMaskedLM.from_pretrained("roberta-base")
mlm.eval(); mlm.cuda()

pairs = datasets.load_from_disk("data/crows")["test"].filter(
    lambda x: x["bias_type"]=="gender")          # isolate gender bias only

def pll(sentence):
    enc = tok(sentence, return_tensors="pt").to("cuda")
    ids = enc["input_ids"][0]
    loss = 0.
    for i in range(1, len(ids)-1):               # leave CLS/SEP intact
        masked = ids.clone()
        masked[i] = tok.mask_token_id
        out = mlm(masked.unsqueeze(0), labels=ids.unsqueeze(0))
        loss += out.loss.item()
    return -loss

correct = 0
for ex in tqdm.tqdm(pairs):
    stereo  = pll(ex["sent_more"])   # more stereotypical
    antist  = pll(ex["sent_less"])   # less stereotypical
    # model is unbiased if it prefers the anti-stereotype version
    if antist >= stereo: correct += 1

acc = correct / len(pairs)
print(f"CrowS-Pairs anti-stereotype preference: {acc:.3f}")
