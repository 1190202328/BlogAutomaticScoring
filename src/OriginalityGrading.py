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
    def get_labels(num):
        i = 1
        labels = list()
        with open("../src/text/urls.txt", "r") as f:
            for line in f.readlines():
                if i > num:
                    break
                i += 1
                line = re.sub("\\t+", "\\t", line)
                labels.append(line.split("\t")[2][:-1])
        return labels

    @staticmethod
    def get_labels_2d(num):
        i = 1
        labels = list()
        with open("../src/text/urls.txt", "r") as f:
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


def train_6d():
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
    labels = OriginalityGrading.get_labels(84)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities()
    k_fold = KFold(n_splits=10, random_state=40, shuffle=True)
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=6))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.2,
                  validation_freq=2)
        # filepath = "../src/saved_model/originality_grading_model.h5"
        # model.save(filepath)
        # model = tf.keras.models.load_model(filepath)
        y_predict = tf.argmax(model.predict(x_test), axis=-1)
        print(classification_report(y_test, y_predict))


def train_2d():
    embedding_len = 1320
    learning_rate = 1e-3
    batch_size = 10
    epochs = 30
    verbose = 2
    opt = tf.optimizers.Adam(learning_rate)
    input_ = tf.keras.Input(shape=(embedding_len,))
    hindden = tf.keras.layers.Dense(64, activation='relu')(input_)
    # hindden = tf.keras.layers.Dense(128, activation='relu')(hindden)
    output = tf.keras.layers.Dense(2, activation='softmax')(hindden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    labels = OriginalityGrading.get_labels_2d(118)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities()
    k_fold = KFold(n_splits=5, random_state=40, shuffle=True)
    for train_index, test_index in k_fold.split(vectors, labels):
        x_train, x_test, y_train, y_test = vectors[train_index], vectors[test_index], labels[train_index], labels[
            test_index]
        y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=2))

        model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.2,
                  validation_freq=2)
        # filepath = "../src/saved_model/originality_grading_model.h5"
        # model.save(filepath)
        # model = tf.keras.models.load_model(filepath)
        y_predict = tf.argmax(model.predict(x_test), axis=-1)
        print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    # train_6d()
    train_2d()
