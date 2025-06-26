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

target4 = r'\[\[ファイル:(.*?)(?:\||\])'
ans4 = re.findall(target4, texts)
for i in range(len(ans4)):
    print(ans4[i])