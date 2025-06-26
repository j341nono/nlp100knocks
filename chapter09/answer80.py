from transformers import AutoTokenizer
import torch
from pprint import pprint

model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

input_text = "The movie was full of incomprehensibilities."

pprint(tokenizer(input_text, return_tensors="pt"))