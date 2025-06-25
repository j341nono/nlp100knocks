from answer30 import processing

def main():
    mapping = processing()
    verb_u = []
    for sentence in mapping:
        for text in sentence:
            if text['pos'] == '動詞':
                verb_u.append(text['base'])
    print(set(verb_u))
if __name__== '__main__':
    main()