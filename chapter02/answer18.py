path = 'data/popular-names.txt'

def main():
    with open(path, 'r') as f:
        line = f.readlines()
        line = map(lambda x: x.replace('\n', ''), line)
        line = list(map(lambda x: x.split('\t'), line))
        output = sorted(line, key=lambda x: int(x[2]), reverse=True)
    print("Displaying 10 items")
    print(output[:10])

if __name__ == '__main__':
    main()
    