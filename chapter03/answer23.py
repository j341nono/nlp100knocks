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

target3 = '(={2,4}.*?={2,4})'
ans3 = re.findall(target3, texts)
for ans in ans3:
    level = ans.count('=') // 2 - 1
    print(ans.replace('=', ''), level)