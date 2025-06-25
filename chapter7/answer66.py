import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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

y_dev_pred = model.predict(X_dev)
cm = confusion_matrix(y_dev, y_dev_pred)

print("混同行列:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("data/confusion_matrix.png")
plt.close()
