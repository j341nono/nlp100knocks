path = "data/popular-names.txt"

with open(path, 'r', encoding='utf-8') as f:
	text = f.read().split('\n')

print(len(text) - 1)
print(text[:3])
