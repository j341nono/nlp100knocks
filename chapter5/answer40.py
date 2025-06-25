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
    {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
    {
        "role": "user",
        "content":
            """
            9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。
            ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
            イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
            ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
            """
    },
]
prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=True
)

output = llm.generate(prompt, sampling_params)

print(output[0].outputs[0].text)