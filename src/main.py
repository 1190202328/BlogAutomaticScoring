# -*- coding:UTF-8 -*-
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
SimilarityCalculator.ge_lda_model("语料库.txt")