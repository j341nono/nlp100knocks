from answer05 import n_gram

char, word = n_gram(2, 'I am an NLPer')
print("文字bi-gram", char)
print("単語bi-gram", word)
print("-"*50)

char, word = n_gram(2, ['I', 'am', 'an', 'NLPer'])
print("文字bi-gram", char)
print("単語bi-gram", word)
