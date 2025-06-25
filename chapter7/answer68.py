import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
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


feature_names = vec.get_feature_names_out()

# モデルの重み (決定関数の特徴量の係数) を取得
weights = model.coef_[0]

weight_feature_pairs = list(zip(weights, feature_names))

top_20_positive = sorted(weight_feature_pairs, key=lambda x: x[0], reverse=True)[:20]

top_20_negative = sorted(weight_feature_pairs, key=lambda x: x[0])[:20]

print("重みの高い特徴量トップ20:")
for i, (weight, feature) in enumerate(top_20_positive, 1):
    print(f"{i}. {feature}: {weight:.4f}")

print("\n重みの低い特徴量トップ20:")
for i, (weight, feature) in enumerate(top_20_negative, 1):
    print(f"{i}. {feature}: {weight:.4f}")