import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle

def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data

def prepare_data(file_path):
    df = pd.read_csv(file_path, sep="\t")
    data = [add_feature(sentence, label) for sentence, label in zip(df["sentence"], df["label"])]
    return df, data

train_path = "data/SST-2/train.tsv"
dev_path = "data/SST-2/dev.tsv"
df_train, data_train = prepare_data(train_path)
df_dev, data_dev = prepare_data(dev_path)

vec = DictVectorizer()
X_train = vec.fit_transform([d["feature"] for d in data_train])
y_train = [d["label"] for d in data_train]
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with open("data/logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("data/vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)


y_train_pred = model.predict(X_train)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)

print("訓練データにおける評価指標:")
print(f"適合率 (Precision): {precision_train:.4f}")
print(f"再現率 (Recall): {recall_train:.4f}")
print(f"F1スコア: {f1_train:.4f}")


y_dev_pred = model.predict(X_dev)
precision_dev = precision_score(y_dev, y_dev_pred)
recall_dev = recall_score(y_dev, y_dev_pred)
f1_dev = f1_score(y_dev, y_dev_pred)

print("\n検証データにおける評価指標:")
print(f"適合率 (Precision): {precision_dev:.4f}")
print(f"再現率 (Recall): {recall_dev:.4f}")
print(f"F1スコア: {f1_dev:.4f}")