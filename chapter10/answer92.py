from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The movie was full of"

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

gen_kwargs = {
    "max_new_tokens": 10,
    "do_sample": True,
    "temperature": 1.0,
    "return_dict_in_generate": True,
    "output_scores": True,
}

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
    )

# 生成されたトークンとその尤度を取得
generated_ids = outputs.sequences[0]
scores = outputs.scores

print("generated token and the score:")
current_text = prompt
for i, (token_id, score) in enumerate(zip(generated_ids[len(input_ids[0]) :], scores)):
    token = tokenizer.decode([token_id])
    probabilities = torch.softmax(score, dim=-1)
    token_prob = probabilities[0, token_id].item()

    # トークンの間に半角スペースを追加
    if not token.startswith(" ") and current_text[-1] != " ":
        current_text += " "
    current_text += token

    print(f"{current_text}")
    print(f"word: {token}, score: {token_prob:.4f}")
    print("-" * 50)