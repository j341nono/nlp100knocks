path = 'data/popular-names.txt'

def main():
    with open(path, 'r') as f:
        line = f.readlines()
        name = map(lambda x: x.split('\t')[0], line)
        name = set(name)
    name = sorted(name)
    print(name)

if __name__ == '__main__':
    main()