#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-31
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np


def get_voc():
    word_list = []
    vectors = []
    idx = 0
    with open('wiki-news-300d-1M-subword.vec') as f:
        f.readline()
        while f:
            words = f.readline().split()
            if not words:
                break
            word = words[0]
            idx += 1
            if idx % 10000 == 0:
                print(idx)
            # print(len(word))
            vector = np.array(words[1:], dtype='float32')
            word_list.append(word)
            vectors.append(vector)
    return word_list, np.array(vectors)


word_list, vectors = get_voc()
word2idx=dict(zip(word_list,range(len(word_list))))
np.save('vectors.npy', vectors)
from joblib import dump
dump(word2idx,'word2idx.pkl')
print(vectors.shape)
