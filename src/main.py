# from datetime import date
import re

from sklearn.model_selection import train_test_split

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.SimilarityCalculator import SimilarityCalculator
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from src.Student import Student
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

students = InfoReader.get_student_info("学生个人博客信息.xlsx")
# path = "./src/text/"
# document_filenames = ["软件构造.txt", "计算机系统.txt", "信息安全.txt"]
# model_related_filename = "最终"
# SimilarityCalculator.train_lsi_model(path, document_filenames, model_related_filename)
# stat_date = date.fromisoformat("2021-04-26")
# end_date = date.fromisoformat("2021-07-08")
# limit = 0.6
# xlsx_name = "博客成绩.xlsx"
# student_score_dict, student_info_dict = BlogAutomaticScoring.calculate_score(students, limit, stat_date, end_date)
# BlogAutomaticScoring.save_scores_to_xlsx(students, student_score_dict, student_info_dict, path, xlsx_name, limit)
stopwords_file = open("./src/text/stopwords.txt")
stopwords_string = stopwords_file.read()
stopwords_file.close()
my_stopwords = stopwords_string.split("\n")
source = open("./src/text/texts.txt", mode="r")
# target_file = open("./src/text/formed_texts.txt", mode="r")
texts = list()
targets = list()
for line in source.readlines():
    if re.match("第(\\d+)篇文章\\[\\d*\\]", line):
        if re.match("第(\\d+)篇文章\\[1\\]", line):
            targets.append(1)
            continue
        targets.append(0)
        continue
    texts.append(line)
clean_texts = SimilarityCalculator.clean(texts, stopwords_set=my_stopwords)
print(len(texts))
print(len(targets))
print(len(clean_texts))
print(texts[11])
print(clean_texts[11])
print(targets[11])
dictionary, corpus_tfidf = SimilarityCalculator.train_tf_idf(clean_texts)
# i = 1
# for items in corpus_tfidf:
#     print("第{}个文档".format(i))
#     i += 1
#     for item in items:
#         print("{0}:{1}".format(dictionary.get(item[0]), item[1]))
#     print("\n")
print(texts[546])
print(clean_texts[546])
print(targets[546])

feature = list()
for items in corpus_tfidf:
    items_feature = [0] * len(dictionary)
    for item in items:
        if dictionary.get(item[0]) is not None:
            items_feature[item[0]] = item[1]
    feature.append(items_feature)
x_train = list()
y_train = list()
x_test = list()
y_test = list()
for i in range(400):
    x_train.append(feature[i])
    y_train.append(targets[i])
lenth = len(targets)
for i in range(400, lenth):
    x_test.append(feature[i])
    y_test.append(targets[i])
print('Start training knn')
knn = KNeighborsClassifier().fit(x_train, y_train)
print('Training done')
answer_knn = knn.predict(x_test)
print('Prediction done')

print('Start training DT')
dt = DecisionTreeClassifier().fit(x_train, y_train)
print('Training done')
answer_dt = dt.predict(x_test)
print('Prediction done')

print('Start training Bayes')
gnb = GaussianNB().fit(x_train, y_train)
print('Training done')
answer_gnb = gnb.predict(x_test)
print('Prediction done')

print('\n\nThe classification report for knn:')
print(classification_report(y_test, answer_knn))
print('\n\nThe classification report for DT:')
print(classification_report(y_test, answer_dt))
print('\n\nThe classification report for Bayes:')
print(classification_report(y_test, answer_gnb))

urls = BlogAutomaticScoring.get_urls(BlogAutomaticScoring.get_main_url("https://blog.csdn.net/weixin_46228614/article"
                                                                       "/details/118551962?spm=1001.2014.3001.5501"))
texts = list()
for url in urls:
    text, _ = BlogAutomaticScoring.get_text(url)
    texts.append(text)
urls = BlogAutomaticScoring.get_urls(BlogAutomaticScoring.get_main_url("https://blog.csdn.net/Louis210?spm=1000.2115"
                                                                       ".3001.5343"))
for url in urls:
    text, _ = BlogAutomaticScoring.get_text(url)
    texts.append(text)
clean_texts = SimilarityCalculator.clean(texts, stopwords_set=my_stopwords)
i = 1
for clean_text in clean_texts:
    print("第{}个文档".format(i))
    print(clean_text)
    i += 1
_, corpus_tfidf = SimilarityCalculator.train_tf_idf(clean_texts, dictionary=dictionary)
feature = list()
for items in corpus_tfidf:
    items_feature = [0] * len(dictionary)
    for item in items:
        if dictionary.get(item[0]) is not None:
            items_feature[item[0]] = item[1]
    feature.append(items_feature)
print(dt.predict(feature))

texts = list()
text, _ = BlogAutomaticScoring.get_text("https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501")
texts.append(text)
clean_texts = SimilarityCalculator.clean(texts, stopwords_set=my_stopwords)
i = 1
for clean_text in clean_texts:
    print("第{}个文档".format(i))
    print(clean_text)
    i += 1
_, corpus_tfidf = SimilarityCalculator.train_tf_idf(clean_texts, dictionary=dictionary)
feature = list()
for items in corpus_tfidf:
    items_feature = [0] * len(dictionary)
    for item in items:
        if dictionary.get(item[0]) is not None:
            items_feature[item[0]] = item[1]
    feature.append(items_feature)
print(dt.predict(feature))
