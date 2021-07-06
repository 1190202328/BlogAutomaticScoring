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
# path = "./src/text/"
# document_filenames = ["软件构造.txt", "计算机系统.txt"]
# model_related_filename = "最终"
# SimilarityCalculator.train_lsi_model(path, document_filenames, model_related_filename)
limit = 0.6
jiang_jing = Student("1190202328", "江经", "https://blog.csdn.net/Louis210/article/details/117415546")
print(BlogAutomaticScoring.calculate_score(jiang_jing, limit))
