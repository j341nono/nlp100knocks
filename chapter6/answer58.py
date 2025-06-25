import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("data/questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[1:5031]
country = list(set(df["v4"].values))

model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)

countryVec = []
countryName = []
for c in country:
    if c in model:
        countryVec.append(model[c])
        countryName.append(c)

# 階層クラスタリング
X = np.array(countryVec)
linkage_result = linkage(X, method="ward", metric="euclidean")

plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor="w", edgecolor="k")
dendrogram(linkage_result, labels=countryName)
plt.tight_layout()
plt.savefig("data/country_dendrogram.png")
plt.close()
