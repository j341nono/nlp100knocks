from transformers import AutoTokenizer, AutoModelForMaskedLM
import itertools
import torch
import torch.nn.functional as F

model_name = "google-bert/bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

input_text = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

encoded = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    output = model(**encoded)
    embeddings = output[0][:, 0, :]

for (i, j) in itertools.combinations(range(len(input_text)), 2):
    sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
    print(f"{input_text[i]} \nand {input_text[j]}: \n{sim:.4f}\n")