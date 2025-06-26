import sys

path = 'data/popular-names.txt'

def main():
    args = sys.argv
    value = int(args[1])
    
    with open(path, 'r') as f:
        line = f.readlines()
        count = len(line) - 1
        for i in reversed(range(value)):
            result = line[count - i].replace('\n', '')
            print(result)

if __name__ == '__main__':
    main()