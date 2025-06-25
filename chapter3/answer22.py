import re
import json
from pprint import pprint

target = 'イギリス'
texts = []
file_path = 'data/jawiki-country.json'

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('title') == target:
            texts = data.get('text')
            break

# () キャプチャグループ
# (?: ...) 非キャプチャグループ、マッチングに使用するが、キャプチャしない

target2 = r'\[\[Category:(.*?)(?:\|.*)?\]\]'
ans2 = re.findall(target2, texts)

pprint(ans2)