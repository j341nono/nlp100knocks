text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'

text = text.replace(',', '').replace('.', '')
text = text.split()

ans = [len(i) for i in text]
ans = [str(num) for num in ans]
print(''.join(ans))
