# 小文字の文字コードは97-122
def cipher(input):
    ans = ''
    for c in input:
        if ord(c) >= 97 and ord(c) <= 122:
            n = 219 - ord(c)
            ans += chr(n)
        else:
            ans += c
    return ans


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