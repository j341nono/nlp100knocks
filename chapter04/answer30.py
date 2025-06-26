from pprint import pprint
path = 'data/neko.txt.mecab'

def processing():
        with open(path, 'r',encoding='utf-8')as f:
            general_list = []
            neko_list = []
            lines = f.readlines()
            for text in lines:
                neko_dic = {}
                suf = text.split("\t")
                if suf[0] == "EOS\n":
                    continue
                temp = suf[1].split(',')
                neko_dic["surface"] = suf[0]
                if len(temp) <= 7:
                    neko_dic["base"] = suf[0]
                else:
                    neko_dic["base"] = temp[6]
                neko_dic["pos"] = temp[0]
                neko_dic["pos1"] = temp[1]
                neko_list.append(neko_dic)
                if suf[0]=="。":
                    general_list.append(neko_list)
                    neko_list = []
        return general_list

def main():
    ans = processing()
    pprint(ans[:2])
    
if __name__ == '__main__':
    main()