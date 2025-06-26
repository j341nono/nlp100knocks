from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding


model_id = "OuteAI/Lite-Oute-1-300M"
train_path = "data/SST-2/train.tsv"
valid_path = "data/SST-2/dev.tsv"


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_csv(train_path, sep="\t", header=0)
    valid_df = pd.read_csv(valid_path, sep="\t", header=0)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    def tokenize_func(examples):
        return tokenizer(
            examples["sentence"],
            padding=True, 
            truncation=True, 
            max_length=512
        )
    
    train_dataset = train_dataset.map(
        tokenize_func, batched=True, remove_columns=["sentence"]
    )
    valid_dataset = valid_dataset.map(
        tokenize_func, batched=True, remove_columns=["sentence"]
    )

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        logging_dir="results/logs",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    valid_results = trainer.evaluate()
    print(f"final results: {valid_results}")


if __name__ == "__main__":
    main()