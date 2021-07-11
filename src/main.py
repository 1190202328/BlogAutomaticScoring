# from datetime import date
from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.SimilarityCalculator import SimilarityCalculator

# students = InfoReader.get_student_info("学生个人博客信息.xlsx")
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

text, _ = BlogAutomaticScoring.get_text(
    "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501")
clean_text = SimilarityCalculator.clean([text])
print(clean_text)
dictionary = SimilarityCalculator.generate_dictionary(texts=clean_text)
print(dictionary)
print(dictionary.get(1))
