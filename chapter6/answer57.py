import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from collections import defaultdict

df = pd.read_csv("data/questions-words.txt", sep=" ", header=None, names=["v1", "v2", "v3", "v4"])
df.dropna(inplace=True) # inplace=True -> 元のdfを直接上書き
df = df.iloc[1:5031]
country = list(set(df["v4"].values))

model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)

countryVec = []
valid_countries = []
for c in country:
    countryVec.append(model[c])
    valid_countries.append(c)

X = np.array(countryVec)
km = KMeans(n_clusters=5, random_state=0)
y_km = km.fit_predict(X)

clusters = defaultdict(list) # キーが存在しない場合に自動で初期値を作る辞書
for country_name, cluster_id in zip(valid_countries, y_km):
    clusters[cluster_id].append(country_name)

for cluster_id, countries in clusters.items():
    print(f"\nCluster {cluster_id} ({len(countries)} countries):")
    print(", ".join(sorted(countries)))
