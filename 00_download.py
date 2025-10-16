# 00_download.py
from datasets import load_dataset

# 1. EDOS Task-A (binary sexism) – splits are on the HF Hub
edos = load_dataset("rewire/edos")             # mirrors the GitHub dump[1]
edos.save_to_disk("data/edos")

# 2. CrowS-Pairs for bias probing
crows = load_dataset("nyu-mll/crows_pairs")    # 1 508 minimal pairs[6]
crows.save_to_disk("data/crows")

print("✔ Datasets cached")
