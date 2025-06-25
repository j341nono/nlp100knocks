path = "data/popular-names.txt"

col1 = "data/col1.txt"
col2 = "data/col2.txt"
col_merge = "data/merge.txt"


with open(col1, 'r') as f1:
    text1 = f1.read().split('\n')
    text1 = text1[:-1]
with open(col2, 'r') as f2:
    text2 = f2.read().split('\n')
    text2 = text2[:-1]

output = []
for i in range(len(text1)):
    output.append(text1[i] + '\t' + text2[i] + '\t')

with open(col_merge, 'w') as f:
    for l in output:
        f.write(l + '\n')


print("saved in "+ col_merge)