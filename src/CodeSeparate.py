from pprint import pprint

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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
    vocab = set()
    f = open("../src/text/代码.txt", "r")
    for line in f.readlines():
        if line != "":
            line = line.replace("\n", "")
            words = line.lower().strip().split(" ")
            sentences.append(words)
            for word in words:
                vocab.add(word)
            labels.append(0)
    f.close()
    f = open("../src/text/文字.txt", "r")
    for line in f.readlines():
        if line != "":
            line = line.replace("\n", "")
            words = line.lower().strip().split(" ")
            sentences.append(words)
            for word in words:
                vocab.add(word)
            labels.append(1)
    f.close()
    return sentences, labels, vocab


sentences, labels, vocab = get_sentences_and_labels()

# 深度学习分类
vocab_list = list(sorted(vocab))
# f = open("../src/text/vocab_list.txt", 'w')
# for vocab in vocab_list:
#     f.write(vocab)
#     f.write("\n")
# f.close()

token_list = []
embedding_len = 50
output_dim = 128
epochs = 3
verbose = 2
vocab_len = len(vocab_list)

for sentence in sentences:
    token = [vocab_list.index(word) for word in sentence]
    token = token[:embedding_len] + [0] * (embedding_len - len(token))
    token_list.append(token)
token_list = np.array(token_list)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(token_list, labels, test_size=0.4, random_state=0)

input_token = tf.keras.Input(shape=(embedding_len,))
embedding = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=output_dim)(input_token)
embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim))(embedding)
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding)
model = tf.keras.Model(input_token, output)
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train, epochs=epochs, verbose=verbose)

path = "../src/saved_model/"
filename = "code_separate_model.h5"
model.save(path+filename)
new_model = tf.keras.models.load_model(path+filename)

y_predict = list()
y_pred = new_model.predict(x_test)
for y in y_pred:
    if y[0] > y[1]:
        y_predict.append(0)
    else:
        y_predict.append(1)
print(classification_report(y_test, y_predict))
