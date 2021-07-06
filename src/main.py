# -*- coding:UTF-8 -*-
from datetime import date
import re

import xlsxwriter as xlsxwriter

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.SimilarityCalculator import SimilarityCalculator

students = InfoReader.get_student_info("学生个人博客信息.xlsx")
for student in students:
    print(student)
print(len(students))
path = "./src/text/"
document_filenames = ["软件构造.txt", "计算机系统.txt", "信息安全.txt"]
model_related_filename = "最终"
SimilarityCalculator.train_lsi_model(path, document_filenames, model_related_filename)
stat_date = date.fromisoformat("2021-04-26")
end_date = date.fromisoformat("2021-07-08")
limit = 0.6
# jiang_jing = Student("1190202328", "江经", "https://blog.csdn.net/Louis210")
# zhu_zhu = Student("1190201924", "朱朱", "https://blog.csdn.net/zhuzyibooooo/article/details/118527726")
# pp = Student("1", "pp", "https://blog.csdn.net/zP1nG/article/details/118418217")
# print(BlogAutomaticScoring.calculate_score(jiang_jing, limit, stat_date, end_date))
# print(BlogAutomaticScoring.calculate_score(zhu_zhu, limit, stat_date, end_date))
# print(BlogAutomaticScoring.calculate_score(pp, limit, stat_date, end_date))
i = 0
workbook = xlsxwriter.Workbook(path+'博客成绩.xlsx')
worksheet = workbook.add_worksheet("sheet1")
worksheet.write(0, 0, "学号")
worksheet.write(0, 1, "姓名")
worksheet.write(0, 2, "url")
worksheet.write(0, 3, "博客成绩")
for student in students:
    if (student.url is None) or (not re.match(".*csdn.*", student.url)):
        i += 1
        print("[{0}]".format(i) + student.__str__() + ":\t0.0")
        worksheet.write(i, 0, student.id)
        worksheet.write(i, 1, student.name)
        worksheet.write(i, 2, student.url)
        worksheet.write(i, 3, 0.0)
        continue
    i += 1
    score = BlogAutomaticScoring.calculate_score(student, limit, stat_date, end_date)
    print("[{0}]".format(i) + student.__str__() + ":\t{0}".format(score))
    worksheet.write(i, 0, student.id)
    worksheet.write(i, 1, student.name)
    worksheet.write(i, 2, student.url)
    worksheet.write(i, 3, score)
workbook.close()