from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="float16",
)

sampling_params = SamplingParams(
    temperature=0.6, top_p=0.9, max_tokens=512, stop="<|eot_id|>"
)


message = [
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。次の質問に回答してください"},
    {
        "role": "user",
        "content": "つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"
    },
]
prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=True
)

output = llm.generate(prompt, sampling_params)

print(output[0].outputs[0].text)