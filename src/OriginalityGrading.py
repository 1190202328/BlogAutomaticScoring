import re

import cv2
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from src.InfoReadAndWrite import InfoReadAndWrite


class OriginalityGrading:
    """
    博客原创性分级类
    """

    @staticmethod
    def get_labels(filepath, num, nd):
        i = 1
        labels = list()
        with open(filepath, "r") as f:
            for line in f.readlines():
                if i > num:
                    break
                i += 1
                line = re.sub("[\\t ]+", "\t", line)
                label = int(line.split("\t")[2].replace("\n", ""))
                if nd == 2:
                    if label < 4:
                        labels.append(0)
                    else:
                        labels.append(1)
                if nd == 3:
                    if label < 3:
                        labels.append(0)
                    elif label < 4:
                        labels.append(1)
                    else:
                        labels.append(2)
                if nd == 6:
                    labels.append(label)
        return labels

    @staticmethod
    def img_show(vector):
        """
        将一个1320维的向量展示为一张灰度图
        :param vector: 1320维的向量，其值在0-1之间
        :return:
        """
        im_array = np.expand_dims(np.reshape(vector, [40, 33]), axis=-1)
        cv2.imshow(" ", cv2.resize(im_array, (1600, 1320)))
        cv2.waitKey(0)


def train_2d_LSTM(data_filepath, label_filepath):
    nd = 2
    filepath = "../src/saved_model/originality_grading_model.h5"
    embedding_len = 1320
    learning_rate = 1e-2
    batch_size = 256
    epochs = 300
    verbose = 2
    validation_freq = 10
    n_splits = 10
    random_state = 30

    opt = tf.optimizers.Adam(learning_rate)
    # checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    # input_ = tf.keras.Input(shape=(embedding_len,))
    input_ = tf.keras.Input(shape=(embedding_len, 1,))
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
    # hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    # hidden = tf.keras.layers.BatchNormalization()(hidden)
    output = tf.keras.layers.Dense(nd, activation='softmax')(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    labels = OriginalityGrading.get_labels(label_filepath, 351, nd)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)
    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []

    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1,
                  validation_freq=validation_freq)
        # , callbacks = [checkpointer]
        # model = tf.keras.models.load_model(filepath)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        print(classification_report(y_test, y_predict))
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def train_2d_logic(data_filepath, label_filepath):
    nd = 2
    filepath = "../src/saved_model/originality_grading_model.h5"
    embedding_len = 1320
    learning_rate = 1e-3
    batch_size = 32
    verbose = 2
    validation_freq = 10
    n_splits = 10
    random_state = 30
    epochs = 2000

    opt = tf.optimizers.Adam(learning_rate)
    # checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    input_ = tf.keras.Input(shape=(embedding_len,))
    hidden = tf.keras.layers.Dense(32, activation='relu')(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    output = tf.keras.layers.Dense(nd, activation='softmax')(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())

    labels = OriginalityGrading.get_labels(label_filepath, 351, nd)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)

    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []

    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1,
                  validation_freq=validation_freq)
        # , callbacks = [checkpointer]
        # model = tf.keras.models.load_model(filepath)
        # print(model.predict(x_train))
        test_loss, test_accuracy = model.evaluate(x_train, y_train)
        print('Accuracy on test_dataset', test_accuracy)

        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        print(classification_report(y_test, y_predict))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def train_3d_logic(data_filepath, label_filepath):
    nd = 3
    filepath = "../src/saved_model/originality_grading_model.h5"
    embedding_len = 1320
    learning_rate = 1e-1
    batch_size = 320
    verbose = 2
    validation_freq = 10
    n_splits = 10
    random_state = 30
    epochs = 500

    opt = tf.optimizers.Adam(learning_rate)
    # checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    input_ = tf.keras.Input(shape=(embedding_len,))
    hidden = tf.keras.layers.Dense(128, activation='relu')(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dense(256, activation='relu')(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    output = tf.keras.layers.Dense(nd, activation='softmax')(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())

    labels = OriginalityGrading.get_labels(label_filepath, 351, nd)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)

    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []

    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1,
                  validation_freq=validation_freq)
        # , callbacks = [checkpointer]
        # model = tf.keras.models.load_model(filepath)
        # print(model.predict(x_train))
        test_loss, test_accuracy = model.evaluate(x_train, y_train)
        print('Accuracy on test_dataset', test_accuracy)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        print(classification_report(y_test, y_predict))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def train_6d_logic(data_filepath, label_filepath):
    nd = 6
    filepath = "../src/saved_model/originality_grading_model.h5"
    embedding_len = 1320
    learning_rate = 1e-3
    batch_size = 32
    verbose = 2
    validation_freq = 10
    n_splits = 10
    random_state = 30
    epochs = 2000

    opt = tf.optimizers.Adam(learning_rate)
    # checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    input_ = tf.keras.Input(shape=(embedding_len,))
    hidden = tf.keras.layers.Dense(256, activation='relu')(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dense(512, activation='relu')(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    output = tf.keras.layers.Dense(nd, activation='softmax')(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())

    labels = OriginalityGrading.get_labels(label_filepath, 351, nd)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)

    k_fold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []

    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.int32(tf.keras.utils.to_categorical(y_train, num_classes=nd))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1,
                  validation_freq=validation_freq)
        # , callbacks = [checkpointer]
        # model = tf.keras.models.load_model(filepath)
        # print(model.predict(x_train))
        test_loss, test_accuracy = model.evaluate(x_train, y_train)
        print('Accuracy on test_dataset', test_accuracy)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        print(classification_report(y_test, y_predict))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def train_ml(data_filepath, label_filepath, nd):
    n_splits = 10
    random_state = 40

    labels = OriginalityGrading.get_labels(label_filepath, 351, nd)
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
        svm = SVC(C=100, gamma=0.001, kernel='rbf', probability=True).fit(x_train, y_train)
        answer_svm = svm.predict(x_test)
        total_y_predict_SVM.extend(answer_svm)

        rf = RandomForestClassifier(n_estimators=90).fit(x_train, y_train)
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


def train_2d_CNN(data_filepath, label_filepath):
    nd = 2
    filepath = "../src/saved_model/originality_grading_model.h5"
    embedding_len = 1320
    learning_rate = 1e-3
    batch_size = 128
    verbose = 2
    validation_freq = 10
    n_splits = 10
    random_state = 30
    epochs = 300

    opt = tf.optimizers.Adam(learning_rate)
    # checkpointer = ModelCheckpoint(filepath=filepath, verbose=verbose, save_best_only=True, monitor='accuracy')
    input_ = tf.keras.Input(shape=[40, 33, 1])
    hidden = tf.keras.layers.Conv2D(32, 3, padding="SAME", activation=tf.nn.relu)(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Conv2D(64, 3, padding="SAME", activation=tf.nn.relu)(hidden)
    hidden = tf.keras.layers.MaxPool2D(strides=[1, 1])(hidden)
    hidden = tf.keras.layers.Conv2D(128, 3, padding="SAME", activation=tf.nn.relu)(hidden)
    flat = tf.keras.layers.Flatten()(hidden)
    hidden = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
    output = tf.keras.layers.Dense(nd, activation=tf.nn.softmax)(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    labels = OriginalityGrading.get_labels(label_filepath, 351, nd)
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
            reshaped_x_train.append(tf.expand_dims(tf.reshape(x_, [40, 33]), axis=-1))
        for x_ in x_test:
            reshaped_x_test.append(tf.expand_dims(tf.reshape(x_, [40, 33]), axis=-1))
        x_train = np.array(reshaped_x_train)
        x_test = np.array(reshaped_x_test)

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_freq=validation_freq, validation_split=0.1)
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
    data_filepath = "../src/text/similarities.csv"
    label_filepath = "../src/text/urls.txt"
    nd = 2
    train_2d_LSTM(data_filepath, label_filepath) #已完成
    train_2d_logic(data_filepath, label_filepath) #已完成
    train_3d_logic(data_filepath, label_filepath)  # 已完成
    train_6d_logic(data_filepath, label_filepath)  # 已完成
    train_ml(data_filepath, label_filepath, nd) #已完成
    train_2d_CNN("../src/text/similarities.csv", "../src/text/urls.txt") #已完成
