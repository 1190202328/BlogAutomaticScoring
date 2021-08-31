import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


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
            f = open("../src/text/vocab_list.txt", 'r')
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
        for sentence in sentences:
            pre_sentence = sentence.lower().strip()
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
        path = "../src/saved_model/"
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
        f = open("../src/text/ptb.train.txt", "r")
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
    k_fold = KFold(n_splits=10, random_state=0, shuffle=True)
    i = 0
    for train_index, test_index in k_fold.split(sentences, labels):
        i += 1
        if i != 5:
            continue
        x_train, x_test, y_train, y_test = sentences[train_index], sentences[test_index], labels[train_index], labels[
            test_index]
        vocab = set()
        for x in x_train:
            for word in x:
                vocab.add(word)
        # 深度学习分类
        vocab_list = list()
        vocab_list.append("<paddle>")
        vocab_list.append("<unk>")
        vocab_list += list(sorted(vocab))
        f = open("../src/text/vocab_list_1.txt", 'w')
        for vocab in vocab_list:
            f.write(vocab)
            f.write("\n")
        f.close()
        # embedding_len = 100
        # output_dim = 64
        # learning_rate = 0.1
        # batch_size = 330
        # epochs = 1
        embedding_len = 50
        output_dim = 32
        learning_rate = 1e-2
        batch_size = 320
        epochs = 2
        verbose = 2
        opt = tf.optimizers.Adam(learning_rate)
        vocab_len = len(vocab_list)
        print("词典大小:{}".format(vocab_len))

        input_token = tf.keras.Input(shape=(embedding_len,))
        embedding = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=output_dim)(input_token)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
        embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim))(embedding)
        embedding = tf.keras.layers.Dense(16, activation=tf.nn.relu)(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
        output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding)
        model = tf.keras.Model(input_token, output)
        model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        print(model.summary())

        x_train = SeparateCode.get_sequences(x_train, embedding_len, vocab_list)
        x_test = SeparateCode.get_sequences(x_test, embedding_len, vocab_list)
        model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, validation_freq=2, verbose=verbose,
                  batch_size=batch_size)
        path = "../src/saved_model/"
        filename = "code_separate_model_1.h5"
        model.save(path + filename)
        model = tf.keras.models.load_model(path + filename)
        y_predict = list()
        y_pred = model.predict(x_test)
        for y in y_pred:
            if y[0] > y[1]:
                y_predict.append(0)
            else:
                y_predict.append(1)
        print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    machine_learning()
