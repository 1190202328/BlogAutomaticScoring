import re
from pprint import pprint

import jieba
import numpy as np
import tensorflow as tf
from baiduspider import BaiduSpider
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.SimilarityCalculator import SimilarityCalculator


class SimilarityFromBERT:
    """
    使用BERT来计算相似度
    """

    @staticmethod
    def get_similarity(sentences):
        print()


if __name__ == '__main__':

    path = "../src/text/"
    fileName = "软件构造博客.txt"
    dictName = "博客数据"
    dictionary = SimilarityCalculator.get_dictionary(path, dictName)


    stopwords_file = open("../src/text/stopwords.txt")
    stopwords_string = stopwords_file.read()
    stopwords_file.close()
    my_stopwords = stopwords_string.split("\n")


    bc = BertClient()
    sentences = ["有一次使用到了contains和indexOf方法", "那什么时候使用Contains的上述方法", " contains方法的源码中其实是使用了IndexOf方法的,但效率还是有差别的",
                 "contains和indexof都可以作为判断是否包含的方法"]
    clean_sentences = SimilarityCalculator.clean_with_low_frequency(sentences, my_stopwords)
    sentences = [" ".join(clean_sentence) for clean_sentence in clean_sentences]
    print(cosine_similarity(bc.encode(sentences)))
