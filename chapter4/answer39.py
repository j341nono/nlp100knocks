from answer30 import processing
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib
import os

save_path = "data/frequency_plot.png"

def main():
    mapping = processing()
    word_list = []
    for sentense in mapping:
        for text in sentense:
            word_list.append(text["surface"])

    data = collections.Counter(word_list)
    temp = sorted(data.values(), reverse=True)

    plt.plot(temp)
    plt.xlabel('出現頻度順位')
    plt.ylabel('出現頻度')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.savefig("data/frequency_plot.png")
    print("saved in " + save_path)
    plt.close()

if __name__ == "__main__":
    main()
