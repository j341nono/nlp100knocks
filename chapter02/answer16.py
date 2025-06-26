import sys

path = 'data/popular-names.txt'


def main():
    arg = sys.argv
    n = arg[1]
    n = int(n)
    
    with open(path, 'r') as f:
        text = f.read().split('\n')
        text = text[:-1]

        part = len(text) // n
        remainder = len(text) % n

        parts = []
        for i in range(n):
            start = i * part
            if i < n - 1 or remainder == 0: # 終了idxを次に
                end = (i + 1) * part
            else:#最後のブロック
                end = len(text)  # 最後の部分を含める
            parts.append(text[start:end])

        for i, part in enumerate(parts):
            part = ''.join(part)
            print(f"Part {i + 1}:\n", part)
    
if __name__ == '__main__':
    main()
    