text_a = 'パトカー'
text_b = 'タクシー'

ans = []

for i, (a, b) in enumerate(zip(text_a, text_b)):
    ans.append(text_a[i])
    ans.append(text_b[i])

ans = ''.join(list(ans))
print(ans)
