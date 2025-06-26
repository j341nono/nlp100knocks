import random

def randomize_sentence(text):
    text = text.split()
    ans = []
    for word in text:
        ans.append(randomize_word(word))
    return ' '.join(ans)

def randomize_word(text):
    if len(text) <= 4:
        return text
    middle = list(text[1:-1])
    random.shuffle(middle)
    return text[0] + ''.join(middle) + text[-1]
    

sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
sentence = randomize_sentence(sentence)

print(sentence)
