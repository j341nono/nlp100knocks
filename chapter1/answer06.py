X_char, _ = n_gram(2, 'paraparaparadise')
Y_char, _ = n_gram(2, 'paragraph')

X = set(X_char)
Y = set(Y_char)

print('和集合:', X|Y)
print('積集合:', X&Y)
print('差集合:')
print(X-Y)
print(Y-X)
