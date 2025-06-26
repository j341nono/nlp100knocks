import collections
from answer30 import processing

# 品詞の種類をファイルから判断
def hinsi(mapping):
    hinsi = []
    for sentence in mapping:
        for i in range(len(sentence)):
            hinsi.append(sentence[i]['pos'])
    return set(hinsi)

def frequency(mapping):
    word = []
    for sentence in mapping:
        for i in range(len(sentence)):
            # if sentence[i]['pos']!='助詞'or sentence[i]['pos']!='連体詞'\
            #     or sentence[i]['pos']!='記号' or sentence[i]['pos']!='助動詞'\
            #     or sentence[i]['pos']!='接続詞':
            #         word.append(sentence[i]['surface'])
            if sentence[i]['pos']=='名詞':
                word.append(sentence[i]['surface'])
    return collections.Counter(word)

def main():
    mapping = processing()
    # print(hinsi(mapping))
    # {'副詞', '動詞', '名詞', '連体詞', '記号', '助動詞', '接続詞', 'フィラー',
    #  '接頭詞', 'その他', '形容詞', '助詞', '感動詞'}
    
    # 単語から除く品詞 --> 助詞,　連体詞, 記号, 助動詞, 接続詞
    c = frequency(mapping)
    print(c.most_common()[:20])
    
if __name__ == '__main__':
    main()