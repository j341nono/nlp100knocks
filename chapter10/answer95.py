"""
2つ目の入力文とその日本語訳
元) Please give me the plural form of the word with its spelling in reverse order.
訳) 単語の複数形を逆順に綴って教えてください。
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token

    message = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please answer the following question.",
        },
        {
            "role": "user", 
            "content": "What do you call a sweet eaten after dinner?"
        },
        {
            "role": "assistant",
            "content": "dessert"
        },
        {
            "role": "user",
            "content": "Please give me the plural form of the word with its spelling in reverse order."
        },
    ]

    prompt = tokenizer.apply_chat_template(message, tokenize=False)
    print("generated prompt:\n")
    print(prompt)
    print("-"*50)

    inputs = tokenizer(
        prompt,
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            do_sample=True,
            temperature=0.2,
            top_k=20, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("genereted text:\n")
    print(response)


if __name__ == "__main__":
    main()