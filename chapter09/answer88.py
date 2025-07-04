import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


model_id = "google-bert/bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained("results")

sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

prediction_dataset = PredictionDataset(sentences, tokenizer)

training_args = TrainingArguments(
    output_dir="results",
    per_device_eval_batch_size=32,
)

trainer = Trainer(
    model=model,
    args=training_args,
)

predictions = trainer.predict(prediction_dataset)
predictions = torch.softmax(torch.tensor(predictions.predictions), dim=1)

for i, sentence in enumerate(sentences):
    print(f"文: {sentence}")
    print(f"肯定的な確率: {predictions[i][1]:.4f}")
    print(f"否定的な確率: {predictions[i][0]:.4f}")
    print(
        "予測ラベル:",
        "肯定的" if predictions[i][1] > predictions[i][0] else "否定的",
    )
    print("-" * 50)