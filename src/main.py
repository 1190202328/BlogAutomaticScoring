from datetime import date
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
path = "./src/text/"
stat_date = date.fromisoformat("2021-04-26")
end_date = date.fromisoformat("2021-07-08")

stopwords_file = open("./src/text/stopwords.txt")
stopwords_string = stopwords_file.read()
stopwords_file.close()
my_stopwords = stopwords_string.split("\n")

source_filename = "texts.txt"
# data, target, dictionary = SimilarityCalculator.generate_dataset(path, source_filename)
dictionary_name = "博客数据"
# dictionary.save(path + dictionary_name + ".dict")
dictionary = SimilarityCalculator.get_dictionary(path, dictionary_name)
filename = "博客数据"
# SimilarityCalculator.save_dataset(path, filename, data, target)
data, target = SimilarityCalculator.load_dataset(path, filename)
# target = label_binarize(target, classes=[0, 1, 3, 5, 6, 9])  # 0或者未标记:软件构造;1:计算机系统;2:信息安全;3:数据结构与算法;4:概率论;5:无效文章6:自学笔记;7:数字逻辑;8:C语言;9:其他课程(将2,4,7,8合并)
# target1 = target[:, 0]
# target2 = target[:, 1]
# target = np.c_[target1, target2]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)

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

# Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3], 'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
# scores = ['precision', 'recall']
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print('%s_weighted' % score)
#     grid = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                         scoring='%s_weighted' % score)  # 1/5作为验证集
#     # 用前一半train数据再做5折交叉验证，因为之前train_test_split已经分割为2份了
#
#     grid_result = grid.fit(x_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print(grid.best_params_)
#     print("Grid scores on development set:")
#
#     means = grid_result.cv_results_['mean_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, param in zip(means, params):
#         print("%f  with:   %r" % (mean, param))
#
#     print("Detailed classification report:")
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     y_true, y_pred = y_test, grid.predict(x_test)
#     print(classification_report(y_true, y_pred))

print('Start training SVM')
classifier = OneVsRestClassifier(SVC(C=10, gamma=0.1, probability=True))
svm = classifier.fit(x_train, y_train)
svm = SVC(C=100, gamma=0.01, probability=True).fit(x_train, y_train)
print('Training done')
answer_svm = svm.predict(x_test)
print('Prediction done')

# model = tensorflow.keras.Sequential([tensorflow.keras.layers.Dense(units=1, input_shape=[10960])])
# model.compile(optimizer='sgd', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=5)
# answer_tf = model.predict(x_test)

print('\n\nThe classification report for knn:')
print(classification_report(y_test, answer_knn))
print('\n\nThe classification report for DT:')
print(classification_report(y_test, answer_dt))
print('\n\nThe classification report for Bayes:')
print(classification_report(y_test, answer_gnb))
print('\n\nThe classification report for SVM:')
print(classification_report(y_test, answer_svm))
# print('\n\nThe classification report for TF :')
# print(classification_report(y_test, answer_tf))
#
# y_score = svm.decision_function(x_test)
# n_classes = 2
# y_score = svm.decision_function(x_test)
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
#     average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])  # 切片，第i个类的分类结果性能
#
# # Compute micro-average curve and area. ravel()将多维数组降为一维
# precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
# average_precision["micro"] = average_precision_score(y_test, y_score,
#                                                      average="micro")
# # Plot Precision-Recall curve for each class
# plt.clf()  # clf 函数用于清除当前图像窗口
# plt.plot(recall["micro"], precision["micro"],
#          label='micro-average Precision-recall curve (area = {0:0.2f})'.format(average_precision["micro"]))
# for i in range(n_classes):
#     plt.plot(recall[i], precision[i],
#              label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
#
# plt.xlim([0.0, 1.05])
# plt.ylim([0.0, 1.05])  # xlim、ylim：分别设置X、Y轴的显示范围。
# plt.xlabel('Recall', fontsize=16)
# plt.ylabel('Precision', fontsize=16)
# plt.title('Extension of Precision-Recall curve to multi-class', fontsize=16)
# plt.legend(loc="lower right")  # legend 是用于设置图例的函数
# plt.show()



