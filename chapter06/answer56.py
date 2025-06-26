import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

def culcCosSim(row):
    global model
    try:
        return model.similarity(row["Word 1"], row["Word 2"])
    except KeyError:
        return None

tqdm.pandas()
model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
df = pd.read_csv("data/wordsim353/combined.csv")
df["cosSim"] = df.progress_apply(culcCosSim, axis=1)

# 欠損値（未知語など）を除いて相関を計算
print(df[["Human (mean)", "cosSim"]].dropna().corr(method="spearman"))
