# -*- coding:UTF-8 -*-
import requests

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.Student import Student
students = InfoReader.get_student_info("学生个人博客信息.xlsx")
for student in students:
    print(student)
print(len(students))

print(BlogAutomaticScoring.get_text(url="https://blog.csdn.net/Louis210/article/details/117415546"))

print(BlogAutomaticScoring.get_urls(main_url="https://blog.csdn.net/Louis210"))