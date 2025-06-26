from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "google-bert/bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

input_sentence = "The movie was full of [MASK]."
input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

with torch.no_grad():
    output = model(input_ids=input_ids)
    logits = output.logits
    
mask_position = input_ids[0].tolist().index(103)

mask_logits = logits[0, mask_position]
top10 = torch.topk(mask_logits, k=10)

top10_ids = top10.indices.tolist()
top10_scores = top10.values.tolist()

predicted_tokens = tokenizer.convert_ids_to_tokens(top10_ids)

for token, score in zip(predicted_tokens, top10_scores):
    print(f"{token}: {score:.4f}")