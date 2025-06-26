from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
import torch


model_id = "OuteAI/Lite-Oute-1-300M"
train_path = "data/SST-2/train.tsv"
valid_path = "data/SST-2/dev.tsv"


def make_prompt(text):
    return f"""Classify the sentiment of the following text as positive or negative. Output only "positive" or "negative".
Text: {text}
Sentiment:"""


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences to the same length
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            # Pad input_ids
            pad_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * pad_length
            padded_input_ids.append(padded_ids)
            
            # Pad labels (use -100 for padded positions to ignore in loss)
            padded_lbls = lbls + [-100] * pad_length
            padded_labels.append(padded_lbls)
            
            # Create attention mask
            mask = [1] * len(ids) + [0] * pad_length
            attention_mask.append(mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predicted_labels = [1 if "positive" in text.lower() else 0 for text in decoded_preds]
    return {"accuracy": accuracy_score(labels, predicted_labels)}


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_csv(train_path, sep="\t", header=0)
    valid_df = pd.read_csv(valid_path, sep="\t", header=0)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    def tokenize_func(examples):
        prompts = [make_prompt(s) for s in examples["sentence"]]
        label_texts = ["positive" if l==1 else "negative" for l in examples["label"]]
        
        # Process each example individually to ensure consistency
        input_ids_list = []
        labels_list = []
        
        for prompt, label_text in zip(prompts, label_texts):
            full_text = prompt + label_text
            
            # Tokenize the full text
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=512,
                padding=False,  # We'll handle padding in the data collator
                return_tensors=None
            )
            
            input_ids_list.append(tokenized["input_ids"])
            # For causal LM, labels are the same as input_ids
            labels_list.append(tokenized["input_ids"][:])  # Copy the list
        
        return {
            "input_ids": input_ids_list,
            "labels": labels_list
        }
    
    train_dataset = train_dataset.map(
        tokenize_func, batched=True, remove_columns=["sentence"]
    )
    valid_dataset = valid_dataset.map(
        tokenize_func, batched=True, remove_columns=["sentence"]
    )

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=64,   # Further reduced batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=64,   # Increased to maintain effective batch size
        eval_accumulation_steps=8,
        logging_dir="results/logs",
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_drop_last=True,
        remove_unused_columns=False,
        logging_steps=100,
    )

    # Use our custom data collator
    data_collator = CustomDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    trainer.train()

    valid_results = trainer.evaluate()
    print(f"final results: {valid_results}")


if __name__ == "__main__":
    main()