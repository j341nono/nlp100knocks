text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
conditions = [1, 5, 6, 7, 8, 9, 15, 16, 19]

text = text.replace(',', '').replace('.', '').split()

ans = {}
for i in range(len(text)):
    if i+1 in conditions:
        ans[text[i][0]] = i+1
    else:
        ans[text[i][:2]] = i+1

print(ans)
