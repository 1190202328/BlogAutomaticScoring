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

# if __name__ == '__main__':


# path = "../src/text/"
# fileName = "文章的标题.txt"
# f = open(path + fileName, mode="w")
#
# urls, titles = BlogAutomaticScoring.get_related_txt("ArrayList与LinkedList的区别", 100)
# for i in range(len(urls)):
#     url = urls[i]
#     print('第{}个网页'.format(i))
#     # _, _, head = BlogAutomaticScoring.get_text(url)
#     print(url)
#     # print(head)
#     print(titles[i])
#     f.write("第{}个标题[1]".format(i+101))
#     f.write("\n")
#     f.write(titles[i])
#     f.write("\n")

# path = "../src/text/"
# fileName = "texts.txt"
# source = open(path + fileName, mode="r")
# output = open(path + "软件构造博客.txt", mode="w")
# pattern = re.compile("\\d+")
# i = 1
# flag = False
# for line in source.readlines():
#     if re.match("第(\\d+)篇文章(\\[)\\d*\\]", line):
#         result = pattern.findall(line)
#         if len(result) == 1:
#             output.write("第{}篇文章[0]".format(i))
#             output.write("\n")
#             i += 1
#             flag = True
#             continue
#         if int(result[1]) == 0:
#             output.write("第{}篇文章[0]".format(i))
#             output.write("\n")
#             i += 1
#             flag = True
#             continue
#         continue
#     if flag:
#         output.write(line)
#         flag = False

# text,_ ,_ = BlogAutomaticScoring.get_text("https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501")
# text = re.sub("\\s+", " ", text)
# f = open("../src/text/江江.txt", mode="w")
# f.write(text)


path = "../src/text/"
fileName = "软件构造博客.txt"
dictName = "博客数据"
dictionary = SimilarityCalculator.get_dictionary(path, dictName)
# source_sentence = "ArrayList与LinkedList的区别"

stopwords_file = open("../src/text/stopwords.txt")
stopwords_string = stopwords_file.read()
stopwords_file.close()
my_stopwords = stopwords_string.split("\n")

# sentences = list()
# # sentences.append(source_sentence)
# targets = list()
# source = open(path + fileName, mode="r")
# pattern = re.compile("\\d+")
# i = 0
# j = 0
# for line in source.readlines():
#     if re.match("第(\\d+)篇文章(\\[)\\d*\\]", line):
#         result = pattern.findall(line)
#         if len(result) == 1:
#             targets.append([0])
#             continue
#         targets.append([int(result[1])])
#         continue
#     sentences.append(line)
# source_sentence = sentences[0]
# clean_sentences = list()
# for sentence in sentences:
#     text = list()
#     for word in jieba.cut(sentence):
#         word = word.lower().strip()
#         text.append(word)
#     clean_sentences.append(text)
# corpus = [dictionary.doc2bow(text) for text in clean_sentences]
# data = list()
# for items in corpus:
#     items_feature = [0] * len(dictionary)
#     for item in items:
#         if dictionary.get(item[0]) is not None:
#             items_feature[item[0]] = item[1]
#     data.append(items_feature)
# similarity1 = cosine_similarity(data)
# limit1 = 0.3
# print(similarity1[0])
# similarity1[0][0] = 0
# similarity1[0][290] = 0
# print(max(similarity1[0]))
# # result1 = list()
# # for i in range(1, len(similarity1[0])):
# #     if similarity1[0][i] > limit1:
# #         result1.append(0)
# #     else:
# #         result1.append(1)
# # print(classification_report(targets, result1))
#
# bc = BertClient()
# vectors = bc.encode(sentences)
# limit2 = 0.92
# similarity2 = cosine_similarity(vectors)
# print(similarity2[0])
# similarity2[0][0] = 0
# similarity2[0][290] = 0
# print(max(similarity2[0]))

# bc = BertClient()
# clean_sentences = SimilarityCalculator.clean_with_low_frequency(sentences, my_stopwords)
# sentences = [" ".join(clean_sentence) for clean_sentence in clean_sentences]
# print(len(sentences))
# print(sentences[0])
# vectors = bc.encode(sentences)
# limit2 = 0.92
# similarity2 = cosine_similarity(vectors)
# print(similarity2[0])
# similarity2[0][0] = 0
# similarity2[0][290] = 0
# print(max(similarity2[0]))

# result2 = list()
# for i in range(1, len(similarity2[0])):
#     if similarity2[0][i] > limit2:
#         result2.append(0)
#     else:
#         result2.append(1)
# print(classification_report(targets, result2))



# bc = BertClient()
# sentences = ["有一次使用到了contains和indexOf方法", "那什么时候使用Contains的上述方法", " contains方法的源码中其实是使用了IndexOf方法的,但效率还是有差别的", "contains和indexof都可以作为判断是否包含的方法"]
# clean_sentences = SimilarityCalculator.clean_with_low_frequency(sentences, my_stopwords)
# sentences = [" ".join(clean_sentence) for clean_sentence in clean_sentences]
# print(cosine_similarity(bc.encode(sentences)))

