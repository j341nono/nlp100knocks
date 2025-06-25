import MeCab

mecab = MeCab.Tagger('')
file_path = 'neko.txt'
save_path = 'neko.txt.mecab'

with open(file_path, 'r', encoding='utf-8') as f1, open(save_path, 'w', encoding='utf-8') as f2:
    line = f1.readlines()
    for text in line:
        ans = mecab.parse(text)
        f2.write(ans)