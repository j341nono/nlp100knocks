path = "data/popular-names.txt"

with open(path, 'r', encoding='utf-8') as f:
    text = f.read().replace('\t', ' ')

print(text[:100])