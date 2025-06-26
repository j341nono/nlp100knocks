import pandas as pd
from gensim.models import KeyedVectors
import torch


sst_train_path = "data/SST-2/train.tsv"
sst_dev_path = "data/SST-2/dev.tsv"

train_df = pd.read_csv(sst_train_path, sep="\t", header=0)
dev_df = pd.read_csv(sst_dev_path, sep="\t", header=0)


vocabulary = set()
for text in train_df["sentence"]:
    words = text.lower().split()
    vocabulary.update(words)

for text in dev_df["sentence"]:
    words = text.lower().split()
    vocabulary.update(words)


model_path = "data/GoogleNews-vectors-negative300.bin"

word_to_id = {"<PAD>": 0}
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

for word in vocabulary:
    if word in model.key_to_index:
        # 次の未使用ID（現在の辞書のサイズ）をIDとして割り当てる
        word_to_id[word] = len(word_to_id)


def process_sst_data(file_path, word_to_id):
    df = pd.read_csv(file_path, sep="\t", header=0)

    processed_data = []
    
    for _, row in df.iterrows():
        # テキストをトークンID列に変換
        tokens = row["sentence"].lower().split()
        input_ids = []
        for token in tokens:
            if token in word_to_id:
                input_ids.append(word_to_id[token])
    
        # 空のトークン列の場合はスキップ
        if not input_ids:
            continue
    
        # データを辞書形式で保存
        data = {
            "text": row["sentence"],
            "label": torch.tensor([float(row["label"])]),
            "input_ids": torch.tensor(input_ids),
        }
        processed_data.append(data)
    return processed_data

train_data = process_sst_data(sst_train_path, word_to_id)
print(f"訓練データ数: {len(train_data)}")

dev_data = process_sst_data(sst_dev_path, word_to_id)
print(f"検証データ数: {len(dev_data)}")


print("\n訓練データのサンプル:")
sample = train_data[0]
print(f"テキスト: {sample['text']}")
print(f"ラベル: {sample['label']}")
print(f"トークンID列: {sample['input_ids']}")