import urllib
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

ans27_dic = {}
for key, value in ans26_dic.items():
    p = r'\[{2}(.*?)\]{2}'
    # r'\1' キャプチャした部分を\1で置換
    ans27_dic[key] = re.sub(p, r'\1', value)
pprint(ans27_dic)

ans28_dic = {}

# ref の削除
for key, value in ans27_dic.items():
    p = r'\<ref.*?\>.*?\</ref\>'
    ans28_dic[key] = re.sub(p, '', value)
pprint(ans28_dic)
# ans28_dic

url = 'https://www.mediawiki.org/w/api.php?action=query&titles=File:' + urllib.parse.quote(ans28_dic['国旗画像']) + '&format=json&prop=imageinfo&iiprop=url'
connection = urllib.request.urlopen(urllib.request.Request(url))
response = json.loads(connection.read().decode())
print(response['query']['pages']['-1']['imageinfo'][0]['url'])