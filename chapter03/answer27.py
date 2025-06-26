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
ans26_dic = {}
for key, value in ans_dic.items():
    ans26_dic[key] = re.sub(r'(\\\{2,5})', '', value)

# [[ ... ]] ...の中身だけを表示
ans27_dic = {}
for key, value in ans26_dic.items():
    p = r'\[{2}(.*?)\]{2}'
    # r'\1' キャプチャした部分を\1で置換
    ans27_dic[key] = re.sub(p, r'\1', value)
pprint(ans27_dic)