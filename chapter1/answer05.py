def n_gram(n, text):
    if isinstance(text, str):
        char = text.replace(" ", "") # stringのみ
        word = text.split(" ")
    elif isinstance(text, list):
        char = ''.join(text)
        word = text

    char_list = []
    word_list = []

    for i in range(len(char) - n+1):
        char_list.append(char[i:n+i])
    for j in range(len(word) - n+1):
        word_list.append(word[j:n+j])
    return char_list, word_list
