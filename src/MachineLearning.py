import re
from pprint import pprint

import jieba
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.SeparateCode import SeparateCode


def get_key_words(path, file):
    key_words = set()
    f = open(path + file, "r")
    for line in f.readlines():
        key_words.add(line.lower().strip().replace("\n", ""))
    f.close()
    return key_words


def get_codes(sentences, key_words):
    predict_labels = [1] * len(sentences)
    codes = list()
    for i in range(len(sentences)):
        for key_word in key_words:
            if key_word in sentences[i]:
                codes.append(sentences[i])
                predict_labels[i] = 0
                break
    return codes, predict_labels


def get_sentences_and_labels():
    sentences = list()
    labels = list()
    f = open("../src/text/扩大的代码.txt", "r")
    for line in f.readlines():
        if line != "":
            line = re.sub("[\\t ]+", " ", line)
            line = line.replace("\n", "")
            words = re.split("[ .]", line.lower().strip())
            # words = jieba.lcut(line.lower().strip())
            sentences.append(words)
            labels.append(0)
    f.close()
    print(len(sentences))
    f = open("../src/text/ptb.txt", "r")
    for line in f.readlines():
        if line != "":
            line = re.sub("[\\t ]+", " ", line)
            line = line.replace("\n", "")
            words = re.split("[ .]", line.lower().strip())
            # words = jieba.lcut(line.lower().strip())
            sentences.append(words)
            labels.append(1)
    f.close()
    print(len(sentences))
    return sentences, labels


sentences, labels = get_sentences_and_labels()
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.3, random_state=0)
vocab = set()
for x in x_train:
    for word in x:
        vocab.add(word)

# 深度学习分类
vocab_list = list()
vocab_list.append("<paddle>")
vocab_list.append("<unk>")
vocab_list += list(sorted(vocab))

# f = open("../src/text/vocab_list.txt", 'w')
# for vocab in vocab_list:
#     f.write(vocab)
#     f.write("\n")
# f.close()

token_list = []
embedding_len = 100
output_dim = 64
learning_rate = 0.1
batch_size = 330
epochs = 1
verbose = 2
vocab_len = len(vocab_list)
print("词典大小:{}".format(vocab_len))

x_train = SeparateCode.get_sequences(x_train, embedding_len, vocab_list)
x_test = SeparateCode.get_sequences(x_test, embedding_len, vocab_list)

input_token = tf.keras.Input(shape=(embedding_len,))
embedding = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=output_dim)(input_token)
embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim))(embedding)
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding)
model = tf.keras.Model(input_token, output)
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

print(model.summary())
print(x_train.shape)
model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, validation_freq=2, verbose=verbose,
          batch_size=batch_size)

path = "../src/saved_model/"
filename = "code_separate_model.h5"
model.save(path+filename)
model = tf.keras.models.load_model(path+filename)

y_predict = list()
y_pred = model.predict(x_test)
for y in y_pred:
    if y[0] > y[1]:
        y_predict.append(0)
    else:
        y_predict.append(1)
print(classification_report(y_test, y_predict))
