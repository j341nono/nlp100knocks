from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math


def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # logits: (batch_size, seq_len, vocab_size)
    # input_ids: (batch_size, seq_len)
    #
    # PyTorchではテンソルのスライスを行った後、内部的に「非連続なメモリレイアウト」になることがある
    # そのまま.view() を使おうとするとエラーが出ることがあるため、
    # 安全のために .contiguous() を使ってメモリを再配置
    # .contiguous() は .view() の前に必ず使う癖をつける
    #
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # クロスエントロピー損失の計算
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # .view() はPyTorchのreshape
    # shift_logits:  (batch_size, seq_len-1, vocab_size) → (batch_size * (seq_len-1), vocab_size)
    # shift_labels:  (batch_size, seq_len-1) → (batch_size * (seq_len-1))
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # 平均損失の計算
    mean_loss = loss.mean()
    # パープレキシティの計算（exp(平均損失)）
    perplexity = math.exp(mean_loss.item())

    return perplexity


model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

sentences = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]

print("each perplexity:")
for sentence in sentences:
    perplexity = calculate_perplexity(model, tokenizer, sentence)
    print(f"sentence: {sentence}")
    print(f"perplexity: {perplexity:.2f}")
    print("-" * 50)