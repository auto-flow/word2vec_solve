#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-11-01
# @Contact    : qichun.tang@bupt.edu.cn
import warnings

import numpy as np
from joblib import load


def cosine_similarity(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)
    return similiarity


class Word2Vec():
    def __init__(self):
        self.word2idx = load('word2idx.pkl')
        self.vectors = np.load('vectors.npy')

    def cosine_dist(self, word1: str, word2: str):
        return cosine_similarity(
            self.fetch_vector(word1),
            self.fetch_vector(word2),
        )

    def cosine_dist2(self, word1: str, word2: str):
        return cosine_similarity(
            self.fetch_vector2(word1),
            self.fetch_vector2(word2),
        )

    def __fetch_vector(self, segment: str):
        if segment not in self.word2idx:
            warnings.warn(f'OOV: {segment}')
            # return np.zeros([300])
            return None
        return self.vectors[self.word2idx[segment], :]

    def fetch_vector(self, word: str):
        word = word.strip()
        if word in self.word2idx:
            return self.__fetch_vector(word)
        if word.lower() in self.word2idx:
            return self.__fetch_vector(word.lower())
        word = word.lower()
        n = len(word)
        k = 3
        vectors = []
        for i in range(n - k + 1):
            sub_vec = self.__fetch_vector(word[i:i + k])
            if sub_vec is not None:
                vectors.append(sub_vec)
        if not vectors:
            warnings.warn(f'All OOV: {word}')
            return np.zeros([300])
        return np.sum(np.array(vectors), axis=0)

    def fetch_vector2(self, word: str):
        # todo: 更多的切分策略，如-，
        words = word.split("_")
        return np.mean([self.__fetch_vector(word) for word in words if word in self.word2idx], axis=0)


word2vec = Word2Vec()
print(word2vec.cosine_dist2('user_name', 'person_name'))
print(word2vec.cosine_dist2('user_name', 'people_name'))
print(word2vec.cosine_dist2('user_name', 'employee_name'))
print(word2vec.cosine_dist2('user_name', 'customer_name'))
print(word2vec.cosine_dist2('user_name', 'commodity_name'))
print(word2vec.cosine_dist2('female', 'woman'))
