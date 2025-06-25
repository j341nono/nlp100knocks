from answer30 import processing

def main():
    mapping = processing()
    ans = []
    for sentence in mapping:
        cnt = 0
        bf = ''
        for i in range(len(sentence)):
            if sentence[i]['pos']=='名詞':
                cnt += 1
                bf += sentence[i]['surface']
            else:
                if cnt >= 2:
                    ans.append(bf)
                cnt = 0
                bf = ''
    print(set(ans))
        
if __name__ == '__main__':
    main()