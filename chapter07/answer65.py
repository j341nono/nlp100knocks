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

df_train = pd.read_csv("data/SST-2/train.tsv", sep="\t")
df_dev = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

data_train = []
for sentence, label in zip(df_train["sentence"], df_train["label"]):
    data_train.append(add_feature(sentence, label))

data_dev = []
for sentence, label in zip(df_dev["sentence"], df_dev["label"]):
    data_dev.append(add_feature(sentence, label))
    
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


first_sentence = df_dev["sentence"].iloc[0]
first_label = df_dev["label"].iloc[0]


predicted_prob = model.predict_proba(X)[0]

print(f"predict: {predicted_prob}")

    
def predict_sentiment(text):
    with open("data/logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    data = add_feature(text)

    X = vec.transform([data["feature"]])

    predicted_label = model.predict(X)[0]
    predicted_prob = model.predict_proba(X)[0]

    sentiment = "ポジティブ" if predicted_label == 1 else "ネガティブ"
    print(f"テキスト: {text}")
    print(f"予測された感情: {sentiment}")
    print(
        f"予測確率: ネガティブ={predicted_prob[0]:.4f}, ポジティブ={predicted_prob[1]:.4f}"
    )

text = "the worst movie I ‘ve ever seen"
predict_sentiment(text)