import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
import numpy as np



def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data


df_train = pd.read_csv("data/SST-2/train.tsv", sep="\t")
df_dev = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

data_train = []
for sentence, label in zip(df_train["sentence"], df_train["label"]):
    data_train.append(add_feature(sentence, label))

data_dev = []
for sentence, label in zip(df_dev["sentence"], df_dev["label"]):
    data_dev.append(add_feature(sentence, label))

    
vec = DictVectorizer(sparse=False)

X_train = vec.fit_transform([d["feature"] for d in data_train])
y_train = [d["label"] for d in data_train]
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

# 正則化パラメータの範囲
C_values = np.logspace(-5, 5, 21)

train_accuracies = []
dev_accuracies = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    dev_pred = model.predict(X_dev)

    train_acc = accuracy_score(y_train, train_pred)
    dev_acc = accuracy_score(y_dev, dev_pred)

    train_accuracies.append(train_acc)
    dev_accuracies.append(dev_acc)

    print(f"C = {C:.2e}, 訓練データの正解率: {train_acc:.4f}, 検証データの正解率: {dev_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_accuracies, "o-", label="train data")
plt.semilogx(C_values, dev_accuracies, "o-", label="dev data")
plt.grid(True)
plt.xlabel("Regularization parameter (C)")
plt.ylabel("accuracy")
plt.title("Regularization parameter and the relationship with accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("data/accuracy_vs_C.png")