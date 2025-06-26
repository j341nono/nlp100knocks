path = "data/popular-names.txt"

col1 = "data/col1.txt"
col2 = "data/col2.txt"

with open(path, 'r') as f:
    text = f.read().split('\n')
    text = text[:-1]

    with open(col1, 'w') as f1, open(col2, 'w') as f2:
        for l in text:
            f1.write(l.split('\t')[0] + '\n')
            f2.write(l.split('\t')[1] + '\n')


print("saved in " + col1 + " and " + col2)