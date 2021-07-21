import re
from pprint import pprint

import jieba
import numpy as np
import tensorflow as tf
from baiduspider import BaiduSpider
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.SimilarityCalculator import SimilarityCalculator

if __name__ == '__main__':
    urls = BlogAutomaticScoring.get_related_txt("git安装", 10)
    for i in range(len(urls)):
        url = urls[i]
        print('第{}个网页'.format(i))
        _, _, head = BlogAutomaticScoring.get_text(url)
        print(url)
        print(head)

    # path = "../src/text/"
    # fileName = "博客数据"
    # dictionary = SimilarityCalculator.get_dictionary(path, fileName)
    # stopwords_file = open("../src/text/stopwords.txt")
    # stopwords_string = stopwords_file.read()
    # stopwords_file.close()
    # my_stopwords = stopwords_string.split("\n")
    # sentences = [
    #     "Java中的List类的contains和indexOf方法的区别",
    #     "java实现contains和indexof方法说明",
    #     "Java中List集合中contains()与indexOf()的区别"
    # ]
    # clean_sentences = list()
    # for sentence in sentences:
    #     text = list()
    #     for word in jieba.cut(sentence):
    #         word = word.lower().strip()
    #         text.append(word)
    #     clean_sentences.append(text)
    # print(clean_sentences)
    # corpus = [dictionary.doc2bow(text) for text in clean_sentences]
    # print(corpus)
    # data = list()
    # for items in corpus:
    #     items_feature = [0] * len(dictionary)
    #     for item in items:
    #         if dictionary.get(item[0]) is not None:
    #             items_feature[item[0]] = item[1]
    #     data.append(items_feature)
    # print(cosine_similarity(data))
    # a = [[1, 3, 2], [2, 2, 1]]
    # print(cosine_similarity(a))
    # b = [[1, 3, 2], [2, 2, 1]]
    # c = [[2, 2, 1]]
    # print(cosine_similarity(b, c))
