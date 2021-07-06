# -*- coding:UTF-8 -*-
from pprint import pprint

import lda as lda
import numpy as np
import requests

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.SimilarityCalculator import SimilarityCalculator
from src.Student import Student

# students = InfoReader.get_student_info("学生个人博客信息.xlsx")
# for student in students:
#     print(student)
# print(len(students))
#
# print(BlogAutomaticScoring.get_text(url="https://blog.csdn.net/Louis210/article/details/117415546"))
#
# doc_clean = SimilarityCalculator.write_clean_corpus_to_file(["软件构造.txt", "计算机系统.txt"], "语料库.txt")
# SimilarityCalculator.get_tfidf_model("语料库.txt")
# documents = SimilarityCalculator.get_documents(["软件构造.txt", "计算机系统.txt"])
# texts = SimilarityCalculator.clean(documents)
# SimilarityCalculator.save_dictionary(texts, "测试词典")
# dictionary = SimilarityCalculator.get_dictionary("测试词典")
# SimilarityCalculator.save_corpus(texts, dictionary, "测试词袋向量")
corpus = SimilarityCalculator.get_corpus("测试词袋向量")
print(corpus)
