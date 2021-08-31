import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from src.InfoReadAndWrite import InfoReadAndWrite


class OriginalityGrading:
    """
    博客原创性分级类
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
                line = re.sub("\\t+", "\\t", line)
                labels.append(line.split("\t")[2][:-1])
        return labels

    @staticmethod
    def get_labels_2d(filepath, num):
        i = 1
        labels = list()
        with open(filepath, "r") as f:
            for line in f.readlines():
                if i > num:
                    break
                i += 1
                line = re.sub("[\\t ]+", "\t", line)
                label = int(line.split("\t")[2].replace("\n", ""))
                if label < 3:
                    labels.append(0)
                else:
                    labels.append(1)
        return labels

    @staticmethod
    def get_labels_3d(filepath, num):
        i = 1
        labels = list()
        with open(filepath, "r") as f:
            for line in f.readlines():
                if i > num:
                    break
                i += 1
                line = re.sub("[\\t ]+", "\t", line)
                label = int(line.split("\t")[2].replace("\n", ""))
                if label < 2:
                    labels.append(0)
                elif label < 4:
                    labels.append(1)
                else:
                    labels.append(2)
        return labels


def train_6d(data_filepath, label_filepath):
    embedding_len = 1320
    learning_rate = 1e-1
    batch_size = 330
    epochs = 10
    verbose = 2
    opt = tf.optimizers.Adam(learning_rate)
    input_ = tf.keras.Input(shape=(embedding_len,))
    hindden = tf.keras.layers.Dense(256, activation='relu')(input_)
    hindden = tf.keras.layers.Dense(128, activation='relu')(hindden)
    output = tf.keras.layers.Dense(6, activation='softmax')(hindden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    labels = OriginalityGrading.get_labels(label_filepath, 351)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)
    k_fold = KFold(n_splits=10, random_state=40, shuffle=True)
    total_y_predict = []
    total_y_test = []
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=6))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.2,
                  validation_freq=2)
        # filepath = "../src/saved_model/originality_grading_model.h5"
        # model.save(filepath)
        # model = tf.keras.models.load_model(filepath)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        print(classification_report(y_test, y_predict))
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def train_2d(data_filepath, label_filepath):
    embedding_len = 1320
    learning_rate = 1e-2
    batch_size = 300
    epochs = 50
    verbose = 2
    opt = tf.optimizers.Adam(learning_rate)
    # input_ = tf.keras.Input(shape=(embedding_len,))
    input_ = tf.keras.Input(shape=(embedding_len, 1,))
    hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))(input_)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    # hidden = tf.keras.layers.Dense(128, activation='relu')(input_)
    # hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    output = tf.keras.layers.Dense(2, activation='softmax')(hidden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    labels = OriginalityGrading.get_labels_2d(label_filepath, 351)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)
    k_fold = KFold(n_splits=10, random_state=30, shuffle=True)
    total_y_predict = []
    total_y_test = []
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=2))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1,
                  validation_freq=2)
        # filepath = "../src/saved_model/originality_grading_model.h5"
        # model.save(filepath)
        # model = tf.keras.models.load_model(filepath)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        print(classification_report(y_test, y_predict))
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


def train_3d(data_filepath, label_filepath):
    embedding_len = 1320
    learning_rate = 1e-4
    batch_size = 30
    epochs = 30
    verbose = 2
    opt = tf.optimizers.Adam(learning_rate)
    input_ = tf.keras.Input(shape=(embedding_len,))
    hindden = tf.keras.layers.Dense(128, activation='relu')(input_)
    hindden = tf.keras.layers.BatchNormalization()(hindden)
    hindden = tf.keras.layers.Dense(32, activation='relu')(hindden)
    hindden = tf.keras.layers.BatchNormalization()(hindden)
    output = tf.keras.layers.Dense(3, activation='softmax')(hindden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    labels = OriginalityGrading.get_labels_3d(label_filepath, 351)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities(data_filepath)
    k_fold = KFold(n_splits=10, random_state=30, shuffle=True)
    total_y_predict = []
    total_y_test = []
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=3))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1,
                  validation_freq=2)
        # filepath = "../src/saved_model/originality_grading_model.h5"
        # model.save(filepath)
        # model = tf.keras.models.load_model(filepath)
        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        print(classification_report(y_test, y_predict))
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


if __name__ == '__main__':
    # train_6d("../src/text/similarities.csv", "../src/text/urls.txt")
    train_2d("../src/text/similarities.csv", "../src/text/urls.txt")
    # train_3d("../src/text/similarities.csv", "../src/text/urls.txt")
