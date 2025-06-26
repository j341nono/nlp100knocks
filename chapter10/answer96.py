from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd


valid_path = "data/SST-2/dev.tsv"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def make_prompt(text):
    return f"""Classify the sentiment of the following text as positive or negative. Output only "positive" or "negative".
Text: {text}
Sentiment:"""


def cal_label(response, prompt):
    generated_text = response[len(prompt):].strip()
    generated_text = generated_text.lower()
    return 1 if "positive" in generated_text else 0

def predict(text, model, tokenizer):
    prompt = make_prompt(text)

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
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cal_label(response, prompt)


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = pd.read_csv(valid_path, sep="\t", header=0)

    correct = 0
    total_count = len(dataset)

    for _, row in dataset.iterrows():
        text = row["sentence"]
        label = row["label"]

        predicted_label = predict(text, model, tokenizer)

        if predicted_label == label:
            correct+=1

    accuracy = correct / total_count
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()