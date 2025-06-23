from answer08 import cipher

print(cipher('b'))
print(cipher('qwerty'))
print("-"*50)

# 暗号化
word ='We are going to meet at 8 at Okaidou'
print(cipher(word))
print("-"*50)

# 復号化
word = 'Wv ziv tlrmt gl nvvg zg 8 zg Opzrwlf'
print(cipher(word))
