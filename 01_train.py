import torch, numpy as np, datasets, os, random
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("🚀 Starting model training...")

# Load dataset
ds = datasets.load_from_disk("data/edos")

# Initialize tokenizer and model
tok = AutoTokenizer.from_pretrained("roberta-base")
def tokenize(batch): 
    return tok(batch["sentence"], truncation=True, padding=False, max_length=192)
ds = ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Training configuration
args = TrainingArguments(
    output_dir="checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
    seed=SEED,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    f1 = f1_score(labels, preds, average="macro")
    accuracy = (preds == labels).mean()
    return {"f1": f1, "accuracy": accuracy}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics
)

print("Training RoBERTa for hate speech detection...")
trainer.train()

# Save final model
trainer.save_model("model")
tok.save_pretrained("model")
print("✅ Model training completed and saved!")
