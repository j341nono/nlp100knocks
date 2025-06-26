from answer30 import processing
from answer35 import frequency
import matplotlib.pyplot as plt
import japanize_matplotlib
import itertools
import collections

def main():
    mapping = processing()
    # c = frequency(mapping)
    
    cat_list = []
    for sentense in mapping:
        text37 = []
        Flag = 0
        for text in sentense:
            if "猫" in text["surface"]:
                Flag = 1
                continue
            # if text["pos"] != "補助記号" and text["pos"] != "助詞" and text["pos"] != "助動詞":
            if text["pos"] == "名詞": # 名詞のみ
                text37.append(text["surface"])
        if Flag == 1:
            cat_list.append(text37)
    all_neko = list(itertools.chain.from_iterable(cat_list))
    c = collections.Counter(all_neko)
    word_list = []
    height_list = []
    for i in range(10):
        word_list.append(c.most_common()[:10][i][0])
        height_list.append(c.most_common()[:10][i][1])
    plt.bar(x = word_list, height = height_list)
    plt.show()
    

if __name__ == "__main__":
    main()