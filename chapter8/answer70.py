import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Tuple

model_path = "data/GoogleNews-vectors-negative300.bin"

model = KeyedVectors.load_word2vec_format(model_path, binary=True)

vocab_size = len(model.key_to_index)
embedding_dim = model.vector_size

# 先頭行にpad用の0ベクトル
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

word_to_id = {"<PAD>": 0}  # パディングトークンのIDは0
id_to_word = {0: "<PAD>"}

# 2行目以降に事前学習済み単語ベクトルを格納
for i, word in enumerate(model.key_to_index, start=1):
    embedding_matrix[i] = model[word]
    word_to_id[word] = i
    id_to_word[i] = word