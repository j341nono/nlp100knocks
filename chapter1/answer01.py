text = 'パタトクカシーー'

ans = []
for i in range(0, 8, 2):
    ans.append(text[i])
# ans = text[::2]

ans = ''.join(list(ans))
print(ans)
