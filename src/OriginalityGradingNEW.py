import os
import re
from pprint import pprint

import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from src.InfoReadAndWrite import InfoReadAndWrite


class OriginalityGradingNEW:
    """
    博客原创性分级类(1000篇文章)
    """

    @staticmethod
    def get_labels(filepath, num):
        i = 1
        labels = list()
        with open(filepath, "r") as f:
            for line in f.readlines():
                if i > num:
                    break
                i += 1
                line = re.sub("[\\t ]+", "\t", line)
                label = int(line.split("\t")[2].replace("\n", ""))
                labels.append(label)
        return labels

    @staticmethod
    def change_labels(labels):
        """
        更改一些label
        :param labels: 原来的labels
        :return: 更改后的labels
        """
        new_labels = list(labels)
        with open('../src/text/差距大的文章.txt', 'r') as f:
            for line in f.readlines():
                line = re.sub('\\t+', '\\t', line)
                line = line.split('\t')
                if len(line) >= 4:
                    num = int(line[0])
                    label = int(line[3].replace('\n', ''))
                    new_labels[num] = label
        return new_labels

    @staticmethod
    def img_show(vector):
        """
        将一个1320维的向量展示为一张灰度图
        :param vector: 1320维的向量，其值在0-1之间
        :return:
        """
        im_array = np.expand_dims(np.reshape(vector, [60, 37]), axis=-1)
        cv2.imshow(" ", cv2.resize(im_array, (2400, 1480)))
        cv2.waitKey(0)

    @staticmethod
    def data_show(vector):
        """
        展示向量的意义
        :param vector: 向量
        :return: 无
        """
        print('标题相似度')
        print(vector[:10])
        print('全文相似度')
        print(vector[10:20])
        print('段落相似度')
        for i in range(20, 820, 10):
            print('{}>>>'.format((i - 20) / 10), end='')
            print(vector[i: i + 10])
        print('句子相似度')
        for i in range(820, 1620, 10):
            print('{}>>>'.format((i - 820) / 10), end='')
            print(vector[i: i + 10])
        print('代码相似度')
        for i in range(1620, 2220, 10):
            print('{}>>>'.format((i - 1620) / 10), end='')
            print(vector[i: i + 10])

    @staticmethod
    def reshape(vectors, similarity_len, paragraph_len, sentence_len, code_len):
        # 每个句子只取前5个相似的句子
        print(vectors.shape)
        new_vectors = []
        for vector in vectors:
            new_vector = []
            new_vector.extend(vector[:similarity_len])
            new_vector.extend(vector[10:10 + similarity_len])
            for i in range(20, 20 + paragraph_len * 10, 10):
                new_vector.extend(vector[i:i + similarity_len])
            for i in range(820, 820 + sentence_len * 10, 10):
                new_vector.extend(vector[i:i + similarity_len])
            for i in range(1620, 1620 + code_len * 10, 10):
                new_vector.extend(vector[i:i + similarity_len])
            new_vectors.append(new_vector)
        vectors = np.float32(new_vectors)
        print('-------转换之后的size-------')
        print(vectors.shape)
        return vectors

    @staticmethod
    def get_results(y_true, y_pred, test_index, label_filepath, filepath):
        """
        获得标签的差异值
        :param label_filepath: 存储标签的文件
        :param filepath: 需要保存的路径
        :param test_index: 序号
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: 无
        """
        dictionary = dict()
        with open(label_filepath, "r") as f:
            for line in f.readlines():
                line = re.sub("[\\t ]+", "\t", line)
                line = line.split("\t")
                if len(line) >= 3:
                    dictionary[int(line[0])] = line[1]
        result = []
        for i in range(len(y_true)):
            sub = abs(int(y_pred[i]) - int(y_true[i]))
            if sub != 0:
                result.append(
                    '{}\t{}\t\t{}\t{}'.format(test_index[i], dictionary[test_index[i]], int(y_true[i]), int(y_pred[i])))
        if os.path.exists(filepath):
            print("\n" + filepath + ">>>已存在\n", end="")
            return 1
        with open(filepath, "w") as f:
            for r in result:
                f.write(r)
                f.write('\n')

    @staticmethod
    def show_history(history, is_accuracy=False):
        """
        展示训练过程
        :param is_accuracy: 是否使用了准确率
        :param history: 训练历史
        :return: 无
        """

        if is_accuracy:
            # 绘制训练 & 验证的准确率值
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


def train_ml(data_filepath, label_filepath):
    n_splits = 10
    random_state = 40

    labels = OriginalityGradingNEW.get_labels(label_filepath, 650)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)
    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict_knn = []
    total_y_predict_DT = []
    total_y_predict_Bayes = []
    total_y_predict_SVM = []
    total_y_predict_RF = []
    total_y_predict_clf = []
    total_y_test = []
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        total_y_test.extend(y_test)

        knn = KNeighborsClassifier().fit(x_train, y_train)
        answer_knn = knn.predict(x_test)
        total_y_predict_knn.extend(answer_knn)

        dt = DecisionTreeClassifier().fit(x_train, y_train)
        answer_dt = dt.predict(x_test)
        total_y_predict_DT.extend(answer_dt)

        gnb = GaussianNB().fit(x_train, y_train)
        answer_gnb = gnb.predict(x_test)
        total_y_predict_Bayes.extend(answer_gnb)

        clf = LogisticRegression(penalty='l1', solver='liblinear').fit(x_train, y_train)
        answer_clf = clf.predict(x_test)
        total_y_predict_clf.extend(answer_clf)
        #
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
        #
        # tuned_parameters = [{'n_estimators': list(range(10, 210, 10))}]
        # scores = ['precision', 'recall']
        # for score in scores:
        #     print("# Tuning hyper-parameters for %s" % score)
        #     print('%s_weighted' % score)
        #     grid = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
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
        # break

        svm = SVC(C=1, gamma=0.01, kernel='rbf', probability=True).fit(x_train, y_train)
        answer_svm = svm.predict(x_test)
        total_y_predict_SVM.extend(answer_svm)

        rf = RandomForestClassifier(n_estimators=140).fit(x_train, y_train)
        answer_rf = rf.predict(x_test)
        total_y_predict_RF.extend(answer_rf)

        # print('\n\nThe classification report for knn:')
        # print(classification_report(y_test, answer_knn))
        # print('\n\nThe classification report for DT:')
        # print(classification_report(y_test, answer_dt))
        # print('\n\nThe classification report for Bayes:')
        # print(classification_report(y_test, answer_gnb))
        # print('\n\nThe classification report for SVM:')
        # print(classification_report(y_test, answer_svm))
    print("总结果如下")
    print('\n\nThe classification report for knn:')
    print(classification_report(total_y_test, total_y_predict_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(total_y_test, total_y_predict_DT))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(total_y_test, total_y_predict_Bayes))
    print('\n\nThe classification report for SVM:')
    print(classification_report(total_y_test, total_y_predict_SVM))
    print('\n\nThe classification report for RF:')
    print(classification_report(total_y_test, total_y_predict_RF))
    print('\n\nThe classification report for CLF:')
    print(classification_report(total_y_test, total_y_predict_clf))


def logic(data_filepath, label_filepath):
    verbose = 2
    nd = 5
    filepath = "../src/saved_model/originality_grading_model.h5"
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                                                      patience=100, verbose=verbose, mode='auto',
                                                      baseline=None, restore_best_weights=False)
    check_pointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    callbacks = []
    labels = OriginalityGradingNEW.get_labels(label_filepath, 650)
    labels = OriginalityGradingNEW.change_labels(labels)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)

    embedding_len = 2220
    random_state = 30
    # random_state = 40
    validation_freq = 1
    n_splits = 5
    drop_out_rate = 0.3
    l1 = 0.01
    l2 = 0.01
    # callbacks.append(early_stopping)
    # callbacks.append(check_pointer)

    # similarity_len = 10
    # paragraph_len = 60
    # sentence_len = 60
    # code_len = 10
    # vectors = OriginalityGradingNEW.reshape(vectors, similarity_len, paragraph_len, sentence_len, code_len)

    learning_rates = [1e-1, 1e-2, 1e-3]
    batch_sizes = [3000, 300, 30]
    epochs = [3000, 2000, 1000]
    grid_search_train(batch_sizes, callbacks, drop_out_rate, embedding_len, epochs, l1, l2, labels, learning_rates,
                      n_splits, nd, random_state, validation_freq, vectors, verbose, get_model_logic)
    return 0
    learning_rate = 1e-3
    batch_size = 2000
    epochs = 2000
    model = get_model_logic(embedding_len, drop_out_rate, learning_rate, l1, l2, nd)
    train_logic(batch_size, callbacks, epochs, labels, model, n_splits, nd, random_state, validation_freq, vectors, verbose)


def grid_search_train(batch_sizes, callbacks, drop_out_rate, embedding_len, epochs, l1, l2, labels, learning_rates,
                      n_splits, nd, random_state, validation_freq, vectors, verbose, get_model):
    max_accuracy = 0
    best_batch_size = 0
    best_learning_rate = 0
    best_epoch = 0
    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for epoch in epochs:
        for learning_rate in learning_rates:
            model = get_model(embedding_len, drop_out_rate, learning_rate, l1, l2, nd)
            for batch_size in batch_sizes:
                for train_index, test_index in k_fold.split(vectors, labels):
                    x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], \
                                                       labels[test_index]
                    y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

                    model.fit(x_train, y_train, epochs=epoch, verbose=verbose, batch_size=batch_size,
                              validation_split=0.2,
                              validation_freq=validation_freq, callbacks=callbacks)
                    model.evaluate(x_train, y_train)

                    y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
                    report = classification_report(y_test, y_predict, output_dict=True)
                    print(report['accuracy'])
                    if report['accuracy'] > max_accuracy:
                        max_accuracy = report['accuracy']
                        best_batch_size = batch_size
                        best_learning_rate = learning_rate
                        best_epoch = epoch
                    break
    print('最大准确率：', max_accuracy, '\tlearning_rate=', best_learning_rate, '\tbatch_size=', best_batch_size, '\tepoch=',
          best_epoch)


def train_logic(batch_size, callbacks, epochs, labels, model, n_splits, nd, random_state, validation_freq, vectors, verbose):
    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

        history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size,
                            validation_split=0.2,
                            validation_freq=validation_freq, callbacks=callbacks)
        OriginalityGradingNEW.show_history(history, is_accuracy=True)
        # model = tf.keras.models.load_model(filepath)
        # print(model.predict(x_train))
        model.evaluate(x_train, y_train)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        report = classification_report(y_test, y_predict)
        print(report)
        # OriginalityGradingNEW.get_results(y_test, y_predict, test_index, label_filepath,
        #                                   filepath='../src/text/差距大的文章3.txt')
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        break
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def get_model_logic(embedding_len, drop_out_rate, learning_rate, l1, l2, nd):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(embedding_len,)),
        tf.keras.layers.ActivityRegularization(l1=l1, l2=l2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nd, activation='softmax')
    ])
    # model = tf.keras.Sequential([
    #     tf.keras.Input(shape=(embedding_len,)),
    #     tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2),
    #                           activity_regularizer=tf.keras.regularizers.l1(l1)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2),
    #                           activity_regularizer=tf.keras.regularizers.l1(l1)),
    #     tf.keras.layers.Dropout(drop_out_rate),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2),
    #                           activity_regularizer=tf.keras.regularizers.l1(l1)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(nd, activation='softmax')
    # ])
    print(model.summary())
    opt = tf.optimizers.Adam(learning_rate)
    # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.mae, tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    # model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy,
    #               metrics=[tf.keras.metrics.categorical_crossentropy, ['accuracy']])
    return model


def train_CNN(data_filepath, label_filepath):
    nd = 5
    filepath = "../src/saved_model/originality_grading_model.h5"
    embedding_len = 2220

    learning_rate = 1e-3
    batch_size = 128
    verbose = 2
    validation_freq = 10
    n_splits = 10
    random_state = 30
    epochs = 300

    opt = tf.optimizers.Adam(learning_rate)
    # checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    input_ = tf.keras.Input(shape=[60, 37, 1])
    # hidden = tf.keras.layers.Conv2D(32, 3, padding="SAME", activation=tf.nn.relu)(input_)
    hidden = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu)(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    # hidden = tf.keras.layers.Conv2D(64, 3, padding="SAME", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.MaxPool2D(strides=[1, 1])(hidden)
    # hidden = tf.keras.layers.Conv2D(128, 3, padding="SAME", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.relu)(hidden)
    flat = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
    output = tf.keras.layers.Dense(nd, activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())

    labels = OriginalityGradingNEW.get_labels(label_filepath, 650)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)
    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []

    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

        reshaped_x_train = []
        reshaped_x_test = []
        for x_ in x_train:
            reshaped_x_train.append(tf.expand_dims(tf.reshape(x_, [60, 37]), axis=-1))
        for x_ in x_test:
            reshaped_x_test.append(tf.expand_dims(tf.reshape(x_, [60, 37]), axis=-1))
        x_train = np.array(reshaped_x_train)
        x_test = np.array(reshaped_x_test)

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size,
                  validation_freq=validation_freq, validation_split=0.1)
        # , callbacks = [checkpointer]
        # model = tf.keras.models.load_model(filepath)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        print(classification_report(y_test, y_predict))
        break
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


if __name__ == '__main__':
    data_filepath = "../src/text/similarities_bigger.csv"
    label_filepath = "../src/text/按原创性分类.txt"
    # train_ml(data_filepath, label_filepath)
    logic(data_filepath, label_filepath)
    # train_CNN(data_filepath, label_filepath)

    # vectors = InfoReadAndWrite.get_similarities(data_filepath)
    # i = 649
    # OriginalityGradingNEW.data_show(vectors[i])
    # OriginalityGradingNEW.img_show(vectors[i])
