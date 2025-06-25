from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
import evaluate


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


model_name = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

train_path = "data/SST-2/train.tsv"
dev_path = "data/SST-2/dev.tsv"

train_df = pd.read_csv(train_path, sep="\t", header=0)
dev_df = pd.read_csv(dev_path, sep="\t", header=0)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(dev_df)

train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["sentence"]
)
val_dataset = val_dataset.map(
    tokenize_function, batched=True, remove_columns=["sentence"]
)

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    logging_dir="results/logs",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()


eval_results = trainer.evaluate()
print(f"最終評価結果: {eval_results}")