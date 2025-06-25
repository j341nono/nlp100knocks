import itertools

path = 'data/popular-names.txt'

def main():
    with open(path, 'r') as f:
        line = f.readlines()
        line = map(lambda x: x.replace('\n', ''), line)
        line = list(map(lambda x: x.split('\t'), line))
        
    line = sorted(line, key=lambda num: num[0]) # 1列目は名前(A-Z)
    grouped = [(k, len(list(g))) for k, g in itertools.groupby(line, lambda num : num[0])] # 名前で
    grouped = sorted(grouped, key=lambda num : num[1], reverse=True)
    for key, count in grouped:
        print(key, count)

if __name__ == '__main__':
    main()