from answer30 import processing

def main():
    mapping = processing()
    noun_tg = []
    for sentence in mapping:
        for i in range(len(sentence)):
            if sentence[i-1]['pos']=='名詞' and sentence[i]['surface']=='の' and sentence[i+1]['pos']=='名詞':
                noun_tg.append(sentence[i-1]['surface'] + sentence[i]['surface'] + sentence[i+1]['surface'])
    print(set(noun_tg))
    
if __name__ == '__main__':
    main()