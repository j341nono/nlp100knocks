import argparse

path = "data/popular-names.txt"

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_arg()
    val = args.n

    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(val):
            lines[i] = lines[i].replace('\n', '')
            print(lines[i])


if __name__=="__main__":
    main()