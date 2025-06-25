import pandas as pd
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# SST-2ファイルの読み込み
sst_train_path = "data/SST-2/train.tsv"
sst_dev_path = "data/SST-2/dev.tsv"

train_df = pd.read_csv(sst_train_path, sep="\t", header=0)
dev_df = pd.read_csv(sst_dev_path, sep="\t", header=0)

# 語彙の構築
vocabulary = set()
for text in pd.concat([train_df["sentence"], dev_df["sentence"]]):
    words = text.lower().split()
    vocabulary.update(words)

# Word2Vecモデルの読み込み
model_path = "data/GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# 単語IDと埋め込み行列の構築
word_to_id = {"<PAD>": 0}
embeddings = [torch.zeros(model.vector_size)]

for word in vocabulary:
    if word in model.key_to_index:
        word_to_id[word] = len(word_to_id)
        embeddings.append(torch.tensor(model[word]))

embedding_matrix = torch.stack(embeddings)


# SST-2のデータをトークンIDに変換
def process_sst_data(file_path, word_to_id):
    df = pd.read_csv(file_path, sep="\t", header=0)
    processed_data = []

    for _, row in df.iterrows():
        tokens = row["sentence"].lower().split()
        input_ids = [word_to_id[token] for token in tokens if token in word_to_id]
        if not input_ids:
            continue
        data = {
            "text": row["sentence"],
            "label": torch.tensor([float(row["label"])]),  # [1] にしておく
            "input_ids": torch.tensor(input_ids),
        }
        processed_data.append(data)

    return processed_data


train_data = process_sst_data(sst_train_path, word_to_id)
dev_data = process_sst_data(sst_dev_path, word_to_id)


# PyTorch Datasetクラス
class SSTDataset(Dataset):
    def __init__(self, data, embedding_matrix):
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"]
        embeddings = self.embedding_matrix[input_ids]
        mean_embedding = torch.mean(embeddings, dim=0)
        return mean_embedding, item["label"].float()


train_dataset = SSTDataset(train_data, embedding_matrix)
dev_dataset = SSTDataset(dev_data, embedding_matrix)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)


# ロジスティック回帰モデル
class LogisticClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


embedding_dim = embedding_matrix.size(1)
model = LogisticClassifier(embedding_dim)

model.load_state_dict(torch.load("data/chapter8/model_epoch10.pth"))


def evaluate(model, dev_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dev_loader:
            outputs = model(inputs)
            prediction = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    acc = 100 * correct / total
    return acc

accuracy = evaluate(model, dev_loader)
print(f"開発セットの正解率: {accuracy:.2f}%")