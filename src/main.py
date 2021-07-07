# -*- coding:UTF-8 -*-
from datetime import date
from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.SimilarityCalculator import SimilarityCalculator

students = InfoReader.get_student_info("学生个人博客信息.xlsx")
# for student in students:
#     print(student)
# print(len(students))
path = "./src/text/"
document_filenames = ["软件构造.txt", "计算机系统.txt", "信息安全.txt"]
model_related_filename = "最终"
SimilarityCalculator.train_lsi_model(path, document_filenames, model_related_filename)
stat_date = date.fromisoformat("2021-04-26")
end_date = date.fromisoformat("2021-07-08")
limit = 0.6
xlsx_name = "博客成绩.xlsx"
student_score_dict, student_info_dict = BlogAutomaticScoring.calculate_score(students, limit, stat_date, end_date)
BlogAutomaticScoring.save_scores_to_xlsx(students, student_score_dict, student_info_dict, path, xlsx_name)
# jiang_jing = Student("1190202328", "江经", "https://blog.csdn.net/Louis210")
# zhu_zhu = Student("1190201924", "朱朱", "https://blog.csdn.net/zhuzyibooooo/article/details/118527726")
# pp = Student("1", "pp", "https://blog.csdn.net/zP1nG/article/details/118418217")
# print(BlogAutomaticScoring.calculate_score(jiang_jing, limit, stat_date, end_date))
# print(BlogAutomaticScoring.calculate_score(zhu_zhu, limit, stat_date, end_date))
# print(BlogAutomaticScoring.calculate_score(pp, limit, stat_date, end_date))