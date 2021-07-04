# -*- coding:UTF-8 -*-
import lda as lda
import requests

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.Student import Student

# students = InfoReader.get_student_info("学生个人博客信息.xlsx")
# for student in students:
#     print(student)
# print(len(students))
#
# print(BlogAutomaticScoring.get_text(url="https://blog.csdn.net/Louis210/article/details/117415546"))
#
# print(BlogAutomaticScoring.get_urls(main_url="https://blog.csdn.net/Louis210"))
#
# print(BlogAutomaticScoring.get_main_url(url="https://blog.csdn.net/Louis210/article/details/117415546"))
# print(BlogAutomaticScoring.get_main_url(url="https://blog.csdn.net/Louis210"))
InfoReader.get_text_from_pdf("/Users/jiangjingjing/Desktop/软件构造/本学期PPT/0.pdf")
