import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import os

df = pd.read_csv("data/questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5030]
country = list(set(df["v4"].values))

model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)

countryVec = []
valid_country = []
for c in country:
    if c in model:
        countryVec.append(model[c])
        valid_country.append(c)

X = np.array(countryVec)

tsne = TSNE(random_state=0, n_iter=15000, metric="cosine")
embs = tsne.fit_transform(X)

plt.figure(figsize=(12, 8))
plt.scatter(embs[:, 0], embs[:, 1])

# 国名ラベルも表示したい場合は以下をコメントアウト解除
# for i, name in enumerate(valid_country):
#     plt.text(embs[i, 0], embs[i, 1], name, fontsize=6)

plt.title("t-SNE of Country Word Embeddings")
plt.tight_layout()
plt.savefig("data/country_tsne.png")
plt.close()
