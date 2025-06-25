from answer30 import processing
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

def main():
    mapping = processing()
    word_list = []
    for sentense in mapping:
        for text in sentense:
            word_list.append(text["surface"])
    data = collections.Counter(word_list)
    plt.hist(data.values(), range(1, 40)) # 40以上はなさそう
    plt.show()
    
if __name__ == "__main__":
    main()