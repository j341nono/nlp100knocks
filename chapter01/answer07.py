def sentence_generator(x, y, z):
    return '{}時の{}は{}'.format(x, y, z)


print(sentence_generator(12, '気温', 22.4))