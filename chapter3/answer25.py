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

target5_ = r'\{\{基礎情報 国\n.*?\n\}\}'
ans_ = re.findall(target5_, texts, re.DOTALL)
ans_ = ''.join(ans_) # list to string

target5 = r'\|(.*?) = (.*?)\n'
ans5 = re.findall(target5, ans_)

ans_dic = dict(ans5)

from pprint import pprint
pprint(ans_dic)