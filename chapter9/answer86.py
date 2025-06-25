import pandas as pd
from transformers import AutoTokenizer
import torch

def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()

def tokenize_texts(texts, tokenizer):
    tokenized_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        tokenized_texts.append(tokens)
    return tokenized_texts

model_id = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_path = "data/SST-2/train.tsv"
dev_path = "data/SST-2/dev.tsv"

train_texts, train_labels = load_data(train_path)
dev_texts, dev_labels = load_data(dev_path)

train_tokenized = tokenize_texts(train_texts, tokenizer)
dev_tokenized = tokenize_texts(dev_texts, tokenizer)

train_tokenized = tokenize_texts(train_texts, tokenizer)
dev_tokenized = tokenize_texts(dev_texts, tokenizer)


sample_texts = train_texts[:4]
sample_labels = train_labels[:4]
sample_tokenized = train_tokenized[:4]


encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")


print("original text:")
for text in sample_texts:
    print(f"- {text}")

print("\ntokens:")
for tokens in sample_tokenized:
    print(f"- {tokens}")

print("\ntoken id after padding:")
print(encoded["input_ids"])

print("\nattension mask:")
print(encoded["attention_mask"])

print("\nlabel:")
print(torch.tensor(sample_labels))