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
