from datasets import load_dataset, Dataset
import pandas as pd
import urllib.request
import os

print("📥 Downloading datasets...")

# 1. Davidson Hate Speech Dataset (for training)
davidson = load_dataset("tdavidson/hate_speech_offensive")["train"]
davidson = davidson.map(lambda x: {
    "sentence": x["tweet"], 
    "label": 1 if x["class"] == 0 else 0  # class=0 is hate_speech
})

# Create train/validation/test splits
davidson = davidson.train_test_split(test_size=0.2, seed=42)
val_test = davidson["test"].train_test_split(test_size=0.5, seed=42)
davidson["validation"] = val_test["train"]
davidson["test"] = val_test["test"]
davidson.save_to_disk("data/edos")

# 2. CrowS-Pairs for bias evaluation
os.makedirs("data/crows", exist_ok=True)
url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
urllib.request.urlretrieve(url, "data/crows/crows_pairs.csv")

crows_df = pd.read_csv("data/crows/crows_pairs.csv")
crows_dataset = Dataset.from_pandas(crows_df)
crows_dataset.save_to_disk("data/crows")

print("✅ Datasets prepared successfully!")
print(f"Training samples: {len(davidson['train'])}")
print(f"CrowS-Pairs samples: {len(crows_df)}")
