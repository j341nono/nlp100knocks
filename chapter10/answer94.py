from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

chat = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Please answer the following question.",
    },
    {
        "role": "user", 
        "content": "What do you call a sweet eaten after dinner?"
    },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False)
print("generated prompt:")
print(prompt)
print("-"*30)


inputs = tokenizer(
    prompt, 
    return_tensors="pt", 
    padding=True, 
    truncation=True, 
    max_length=512
)

outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs["attention_mask"],
    max_new_tokens=60,
    do_sample=True,
    temperature=0.7,
    top_k=50, 
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("genereted text:")
print(response)