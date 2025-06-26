import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# モデルとトークナイザの準備
model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="float16",
)

df = pd.read_csv("JMMLU/JMMLU/japanese_history.csv", header=None)
df.columns = ["問題", "選択肢A", "選択肢B", "選択肢C", "選択肢D", "正解"]

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.9, max_tokens=512, stop="<|eot_id|>"
)

def create_prompt(question, a, b, c, d):
    return [
        {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
        {
            "role": "user",
            "content": f"{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n正しい選択肢を一つだけアルファベットで答えてください。"
        }
    ]

def llm_score(sampling_params=sampling_params):
    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        messages = create_prompt(row["問題"], row["選択肢A"], row["選択肢B"], row["選択肢C"], row["選択肢D"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output = llm.generate(prompt,sampling_params) 
        answer = output[0].outputs[0].text.strip()

        if answer.upper().startswith(row["正解"].strip().upper()):
            correct += 1

    accuracy = correct / total * 100
    print(f"正解数: {correct} / {total} = {accuracy:.2f}%")

for temp in ([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
    sampling_params = SamplingParams(
        temperature=temp, top_p=0.9, max_tokens=512, stop="<|eot_id|>"
    )
    print(f"temperature: ", temp)
    llm_score(sampling_params)