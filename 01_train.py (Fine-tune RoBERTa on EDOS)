# 01_train.py
import torch, numpy as np, evaluate, datasets, os, random
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ds = datasets.load_from_disk("data/edos")      # <- cached in step 0
ds = ds.rename_column("text", "sentence")
ds = ds.map(lambda x: {"label": 0 if x["label"]=="not sexist" else 1})

tok = AutoTokenizer.from_pretrained("roberta-base")
def tokenize(batch): return tok(batch["sentence"],
                                truncation=True, padding=False, max_length=192)
ds = ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base",
                                                           num_labels=2)

args = TrainingArguments(
    "checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    seed=SEED,
)

metric = evaluate.load("f1")
def compute(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"f1": metric.compute(predictions=preds, references=labels,
                                 average="macro")["f1"]}

trainer = Trainer(model, args,
                  train_dataset=ds["train"],
                  eval_dataset=ds["validation"],
                  tokenizer=tok,
                  data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=compute)

trainer.train()
trainer.save_model("model")
tok.save_pretrained("model")
print("✔ Model saved to model/")
