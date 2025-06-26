from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "google-bert/bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

# MASKトークンIDを確認
print("mask_token: ", tokenizer.mask_token)
print("mask_token_id: ", tokenizer.mask_token_id)


input_sentence = "The movie was full of [MASK]."
input_ids = tokenizer.encode(input_sentence, return_tensors="pt")

with torch.no_grad():
    output = model(input_ids=input_ids)
    logits = output.logits
    
mask_position = input_ids[0].tolist().index(103)

# 貪欲に選択
id_best = logits[0, mask_position].argmax(-1).item()
token_best = tokenizer.convert_ids_to_tokens(id_best)

print(token_best)