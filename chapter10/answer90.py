from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The movie was full of"

inputs = tokenizer(prompt, return_tensors="pt")
print("tokened prompt:")
print(f"トークンID: {inputs['input_ids'][0]}")
print(
    f"トークン: {[tokenizer.decode([token_id]) for token_id in inputs['input_ids'][0]]}"
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(logits, dim=0)

k = 10
top_k = torch.topk(probabilities, k=k)
top_indices = top_k.indices
top_probabilities = top_k.values

print("\npredicted token / proba:")
for i in range(k):
    token = tokenizer.decode([top_indices[i]])
    probability = top_probabilities[i].item()
    print(f"{i + 1}. token: {token}, proba: {probability:.4f}")