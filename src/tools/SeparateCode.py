import re
from pprint import pprint

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split

from src.tools import Clean
from src.machine_learning import data_analysis


class SeparateCode:
    """
    用于将文本和代码分离的类
    """
    chinese_pattern = r'[\u4e00-\u9fa5].*[\u4e00-\u9fa5]'

    @staticmethod
    def clean_code_line(code_line, low_limit=7, high_limit=30):
        """
        获得干净的code(不含\t，\n，连续2个以上空格，注释)
        目前注释只支持
        多行注释：/**/和三个'和三个"
        单行注释：//和#
        :param high_limit: 每行代码的最长长度，大于该长度的代码行将会被过滤
        :param low_limit: 每行代码的最短长度，小于该长度的代码行将会被过滤
        :param code_line: 源代码行
        :return: 干净的code
        """
        blank_pattern = "[\\t\\n]+"
        code_line = re.sub("/(\\*).*?(\\*)/", "", code_line, flags=re.S)
        code_line = re.sub("'''.*?'''", "", code_line, flags=re.S)
        code_line = re.sub('""".*?"""', "", code_line, flags=re.S)
        code_line = re.sub("\n+", "", code_line)
        code_line = re.sub(blank_pattern, "", code_line)
        code_line = re.sub(" +", " ", code_line)
        java_start = code_line.find("//")
        python_start = code_line.find("#")
        if java_start != -1:
            code_line = code_line[0:java_start]
        if python_start != -1:
            code_line = code_line[0:python_start]
        if len(code_line) < low_limit or len(code_line) > high_limit:
            return ""
        return code_line

    @staticmethod
    def get_sequences(sentences, embedding_len, vocab_list=None):
        """
        获取句子对应的词序列
        :param sentences: 句子（一行为一个句子）（每个句子为一个列表，列表中每个元素为一个词）
        :param embedding_len: 词向量维度
        :param vocab_list: 词典列表
        :return: 词序列
        """
        if not vocab_list:
            vocab_list = list()
            f = open("../../text/vocab_list.txt", 'r')
            # f = open("../src/text/vocab_list_1.txt", 'r')
            for line in f.readlines():
                vocab_list.append(line[:-1])
            f.close()
        token_list = list()
        for sentence in sentences:
            token = list()
            i = 0
            for word in sentence:
                if i >= embedding_len:
                    break
                i += 1
                if word not in vocab_list:
                    token.append(1)
                else:
                    token.append(vocab_list.index(word))
            token = token[:embedding_len] + [0] * (embedding_len - len(token))
            token_list.append(token)
        token_list = np.array(token_list)
        return token_list

    @staticmethod
    def get_codes(text):
        """
        根据文章分离出codes
        :param text: 文章
        :return: codes的列表
        """
        codes = list()
        embedding_len = 100
        # embedding_len = 50
        sentences = text.split("\n")
        code_like_sentences_list = list()
        code_like_sentences = list()
        sentences_flag = list()
        clean_sentences = []
        for sentence in sentences:
            if sentence == "" or re.match("\\s+", sentence):
                continue
            clean_sentences.append(sentence)
        sentences = clean_sentences
        for sentence in sentences:
            pre_sentence = sentence.lower().strip()
            if re.match("[\\w\\.]*print.*", pre_sentence):
                sentences_flag.append(0)
                continue
            pre_sentence = SeparateCode.clean_code_line(pre_sentence, low_limit=7, high_limit=100)
            if pre_sentence == "" or re.match('[_. ]+', pre_sentence) or re.match("((https)|(http)).*", pre_sentence) \
                    or re.match('(\\d+)(\\.\\d+)*.*', pre_sentence) or re.match(r'[\u4e00-\u9fa5]+', pre_sentence) \
                    or re.match('((\\(\\d+\\))|(（\\d+）)).*', pre_sentence):
                sentences_flag.append(1)
                continue
            if re.match(r'.*[\u4e00-\u9fa5]+.*', pre_sentence):
                if not re.match(r'([^\u4e00-\u9fa5]+".*?"[^\u4e00-\u9fa5]+)+', pre_sentence):
                    sentences_flag.append(1)
                    continue
            sentences_flag.append(0)
            # code_like_sentences.append(sentence)
            # pre_sentence = re.split("[ .]", pre_sentence)
            # code_like_sentences_list.append(pre_sentence)
        # pprint(sentences)
        for i in range(1, len(sentences_flag) - 1):
            if sentences_flag[i - 1] == 1 and sentences_flag[i] == 0 and sentences_flag[i + 1] == 1:
                sentences_flag[i] = 1
        for i in range(1, len(sentences_flag) - 2):
            if sentences_flag[i - 1] == 1 and sentences_flag[i] == 0 and sentences_flag[i + 1] == 0 and sentences_flag[
                i + 2] == 1:
                sentences_flag[i] = 1
                sentences_flag[i + 1] = 1
        for i in range(len(sentences_flag)):
            if sentences_flag[i] == 0:
                code_like_sentences.append(sentences[i])
                pre_sentence = sentences[i].lower().strip()
                pre_sentence = SeparateCode.clean_code_line(pre_sentence, low_limit=7, high_limit=100)
                pre_sentence = re.split("[ .]", pre_sentence)
                code_like_sentences_list.append(pre_sentence)
        if not code_like_sentences:
            return codes
        # print("可能是代码的如下(共{}个):".format(len(code_like_sentences)))
        # i = 1
        # for code_like_sentence in code_like_sentences:
        #     print("[{}]>>>".format(i) + code_like_sentence)
        #     i += 1
        code_indexes = [1] * len(code_like_sentences_list)
        sequences = SeparateCode.get_sequences(code_like_sentences_list, embedding_len)
        path = "../saved_model/"
        filename = "code_separate_model.h5"
        # filename = "code_separate_model_1.h5"
        model = tf.keras.models.load_model(path + filename)
        rs = model.predict(sequences)
        for i in range(len(rs)):
            if rs[i][0] > rs[i][1]:
                code_indexes[i] = 0
        for i in range(1, len(code_indexes) - 1):
            if code_indexes[i] == 1 and code_indexes[i - 1] == 0 and code_indexes[i + 1] == 0:
                code_indexes[i] = 0
        for i in range(len(code_indexes)):
            if code_indexes[i] == 0:
                codes.append(code_like_sentences[i])
        i = 1
        # print("检测出的代码的如下(共{}个):".format(len(codes)))
        # for code in codes:
        #     print("[{}]>>>".format(i) + code)
        #     i += 1
        return codes

    @staticmethod
    def get_key_words(path, file):
        key_words = set()
        f = open(path + file, "r")
        for line in f.readlines():
            key_words.add(line.lower().strip().replace("\n", ""))
        f.close()
        return key_words

    @staticmethod
    def get_sentences_and_labels():
        sentences = list()
        labels = list()
        f = open("../../text/扩大的代码.txt", "r")
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
        f = open("../../text/ptb.train.txt", "r")
        i = 0
        for line in f.readlines():
            if line != "":
                if i > 14000:
                    break
                i += 1
                line = re.sub("[\\t ]+", " ", line)
                line = line.replace("\n", "")
                words = re.split("[ .]", line.lower().strip())
                # words = jieba.lcut(line.lower().strip())
                sentences.append(words)
                labels.append(1)
        f.close()
        print(len(sentences))
        return sentences, labels


def machine_learning():
    sentences, labels = SeparateCode.get_sentences_and_labels()
    labels = np.array(labels)
    sentences = np.array(sentences, dtype=object)
    random_state = 40
    sentences, x_valid, labels, y_valid = train_test_split(sentences, labels, test_size=0.2, shuffle=True,
                                                           random_state=random_state)
    k_fold = KFold(n_splits=5, random_state=random_state, shuffle=True)
    total_y_predict = []
    total_y_test = []
    # i = 0
    for train_index, test_index in k_fold.split(sentences, labels):
        # i += 1
        # if i != 5:
        #     continue
        x_train, x_test, y_train, y_test = sentences[train_index], sentences[test_index], labels[train_index], labels[
            test_index]
        vocab = set()
        for x in x_train:
            for word in x:
                vocab.add(word)
        # 深度学习分类
        vocab_list = list()
        # vocab_list.append("<paddle>")
        # vocab_list.append("<unk>")
        # vocab_list += list(sorted(vocab))

        # f = open("../src/text/vocab_list_1.txt", 'w')
        # for vocab in vocab_list:
        #     f.write(vocab)
        #     f.write("\n")
        # f.close()

        f = open("../../text/vocab_list.txt", 'r')
        for line in f.readlines():
            vocab_list.append(line[:-1])
        f.close()

        # embedding_len = 100
        # output_dim = 64
        # learning_rate = 0.1
        # batch_size = 330
        # epochs = 1
        embedding_len = 100
        # embedding_len = 50
        output_dim = 32
        learning_rate = 1e-5
        batch_size = 32
        epochs = 20
        verbose = 1
        l1 = 0
        l2 = 1e-2
        drop_out_rate = 1e-3
        opt = tf.optimizers.Adam(learning_rate)
        vocab_len = len(vocab_list)
        print("词典大小:{}".format(vocab_len))

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(embedding_len,)),
            tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=output_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ActivityRegularization(l1=l1, l2=l2),
            tf.keras.layers.Dropout(drop_out_rate),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        print(model.summary())

        # log_dir = '../src/logs/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        call_backs = []
        # call_backs.append(tensorboard_callback)

        x_train = SeparateCode.get_sequences(x_train, embedding_len, vocab_list)
        x_test = SeparateCode.get_sequences(x_test, embedding_len, vocab_list)
        x_valid_sequences = SeparateCode.get_sequences(x_valid, embedding_len, vocab_list)
        # history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid_sequences, y_valid),
        #                     verbose=verbose,
        #                     batch_size=batch_size, callbacks=call_backs)
        # data_analysis.show_history(history, is_accuracy=True)

        path = "../saved_model/"
        filename = "code_separate_model.h5"
        # model.save(path + filename)
        model = tf.keras.models.load_model(path + filename)

        y_predict = list(tf.argmax(model.predict(x_test), axis=-1))
        print(classification_report(y_test, y_predict))
        total_y_predict.extend(y_predict)
        total_y_test.extend(y_test)
        break
    print("总结果")
    print(classification_report(total_y_test, total_y_predict))


if __name__ == '__main__':
    machine_learning()

    # f = open("../src/text/江江.txt")
    # text = f.read()
    # f.close()
    # print(text)

    # f = open("../../text/代码分离测试.txt")
    # text = f.read()
    # f.close()
    # print(text)
    #
    # text = re.sub("(\\xa0)|(\\u200b)|(\\u2003)|(\\u3000)", "", text)
    # text = re.sub("[\\t ]+", " ", text)
    # text = re.sub("\n+", "\n", text)
    # text = re.sub("(\n +)|( +\n)", "\n", text)
    # to_search_code_text = Clean.clean_code_for_text(text)
    # print('--------------------------------')
    # print(to_search_code_text)
    # pprint(SeparateCode.get_codes(to_search_code_text))
