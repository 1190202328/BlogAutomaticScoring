from datetime import date

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.InfoReader import InfoReader
from src.SimilarityCalculator import SimilarityCalculator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

students = InfoReader.get_student_info("学生个人博客信息.xlsx")
path = "../src/text/"

stopwords_file = open("../src/text/stopwords.txt")
stopwords_string = stopwords_file.read()
stopwords_file.close()
my_stopwords = stopwords_string.split("\n")

source_filename = "../src/text/texts.txt"
# data, target, dictionary = SimilarityCalculator.generate_dataset(path, source_filename)
dictionary_name = "博客数据"
# dictionary.save(path + dictionary_name + ".dict")
dictionary = SimilarityCalculator.get_dictionary(path, dictionary_name)
filename = "博客数据"
# SimilarityCalculator.save_dataset(path, filename, data, target)
data, target = SimilarityCalculator.load_dataset(path, filename)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)

print('Start training knn')
knn = KNeighborsClassifier().fit(x_train, y_train)
print('Training done')
answer_knn = knn.predict(x_test)
print('Prediction done')

related_urls = BlogAutomaticScoring.get_course_related_urls(students, knn, dictionary)
f = open("../src/text/url总集.txt", "w")
for related_url in related_urls:
    f.write(related_url)
    f.write("\n")
f.close()
