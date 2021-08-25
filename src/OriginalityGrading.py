import re
from pprint import pprint

import jieba
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.InfoReadAndWrite import InfoReadAndWrite
from src.SeparateCode import SeparateCode


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


if __name__ == '__main__':
    labels = OriginalityGrading.get_labels(60)
    labels = np.array(labels, dtype=int)
    vectors = InfoReadAndWrite.get_similarities()

    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=0)
    y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=6))

    embedding_len = 1320
    learning_rate = 1e-1
    batch_size = 330
    epochs = 10
    verbose = 2

    opt = tf.optimizers.Adam(learning_rate)
    input_ = tf.keras.Input(shape=(embedding_len,))
    # hidden = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(embedding_len))(input_)
    hindden = tf.keras.layers.Dense(256, activation='relu')(input_)
    hindden = tf.keras.layers.Dense(128, activation='relu')(hindden)
    output = tf.keras.layers.Dense(6, activation='softmax')(hindden)
    model = tf.keras.Model(input_, output)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.1, validation_freq=2)

    filepath = "../src/saved_model/originality_grading_model.h5"
    model.save(filepath)
    model = tf.keras.models.load_model(filepath)

    y_predict = tf.argmax(model.predict(x_test), axis=-1)
    print(classification_report(y_test, y_predict))
