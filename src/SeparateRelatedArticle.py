import json
import re
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.Pretreatment import Pretreatment


class SeparateRelatedArticle:
    """
    将文章列表分离出软讲座主题相关文章的工具类
    """

    @staticmethod
    def get_urls(filepath):
        """
        从磁盘中读如urls
        :param filepath: 文件路径
        :return: urls列表
        """
        urls = list()
        f = open(filepath, "r")
        for line in f.readlines():
            r = line.split("\t")
            urls.append(r[1].replace("\n", " "))
        f.close()
        return urls

    @staticmethod
    def write_article_to_file(urls, filepath):
        """
        将urls列表对应的文章都写入文件
        :param filepath:
        :return:
        """
        results = list()
        i = 1
        for url in urls:
            print("正在处理第{}个url(共{}个)>>>".format(i, len(urls)) + url)
            i += 1
            results.append(Pretreatment.split_txt(url, EDU=False, verbose=False))
        with open(filepath, 'w') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=2))

    @staticmethod
    def get_results(filepath):
        """
        获得根据urls得到的results
        :return: results
        """
        with open(filepath, 'r') as f:
            return json.loads(f.read())

    @staticmethod
    def get_sequences(texts, embedding_len, vocab_list=None):
        """
        获取句子对应的词序列
        :param texts: 文章列表
        :param embedding_len: 每个向量维度
        :param vocab_list: 词典列表
        :return: 词序列
        """
        if not vocab_list:
            vocab_list = list()
            f = open("../src/text/vocab_list_text.txt", 'r')
            for line in f.readlines():
                vocab_list.append(line[:-1])
            f.close()
        token_list = list()
        for text in texts:
            i = 0
            token = list()
            for word in text:
                if i > embedding_len:
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
    def get_texts_and_labels():
        texts = list()
        labels = list()
        source = open("../src/text/texts.txt", mode="r")
        pattern = re.compile("\\d+")
        for line in source.readlines():
            if re.match("第(\\d+)篇文章\\[\\d*\\]", line):
                result = pattern.findall(line)
                if len(result) == 1:
                    labels.append(0)
                else:
                    if int(result[1]) == 0:
                        labels.append(0)
                    else:
                        labels.append(1)
            else:
                texts.append(line)
        return texts, labels

    @staticmethod
    def get_course_related_urls(urls):
        vocab_list = list()
        f = open("../src/text/vocab_list_text.txt", 'r')
        for line in f.readlines():
            vocab_list.append(line[:-1])
        f.close()
        valid_urls = list()
        course_related_urls = list()
        texts = list()
        embedding_len = 200
        for url in urls:
            result = Pretreatment.split_txt(url, False, verbose=False)
            if result and result.get('text'):
                valid_urls.append(url)
                texts.append(result.get('text'))
        texts = Pretreatment.clean_with_low_frequency(texts)
        sequences = SeparateRelatedArticle.get_sequences(texts, embedding_len, vocab_list=vocab_list)
        model = tf.keras.models.load_model("../src/saved_model/ralated_text_separate_model.h5")
        y_pred = model.predict(sequences)
        for i in range(len(y_pred)):
            if y_pred[i][0] > y_pred[i][1]:
                course_related_urls.append(valid_urls[i])
        return course_related_urls


if __name__ == '__main__':
    # texts, labels = SeparateRelatedArticle.get_texts_and_labels()
    # labels = np.array(labels)
    # texts = Pretreatment.clean_with_low_frequency(texts)
    # x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=0)
    # vocab = set()
    # for x in x_train:
    #     for word in x:
    #         vocab.add(word)
    #
    # # 深度学习分类
    # vocab_list = list()
    # vocab_list.append("<paddle>")
    # vocab_list.append("<unk>")
    # vocab_list += list(sorted(vocab))
    #
    # f = open("../src/text/vocab_list_text.txt", 'w')
    # for vocab in vocab_list:
    #     f.write(vocab)
    #     f.write("\n")
    # f.close()
    #
    # token_list = []
    # embedding_len = 200
    # output_dim = 64
    # batch_size = 128
    # epochs = 10
    # verbose = 2
    # vocab_len = len(vocab_list)
    # print("词典大小:{}".format(vocab_len))
    #
    # x_train = SeparateRelatedArticle.get_sequences(x_train, embedding_len, vocab_list)
    # x_test = SeparateRelatedArticle.get_sequences(x_test, embedding_len, vocab_list)
    #
    # input_token = tf.keras.Input(shape=(embedding_len,))
    # embedding = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=output_dim)(input_token)
    # embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim))(embedding)
    # embedding = tf.keras.layers.Dropout(0.5)(embedding)
    # output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding)
    # model = tf.keras.Model(input_token, output)
    # model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    #
    # print(model.summary())
    # model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, validation_freq=2, verbose=verbose,
    #           batch_size=batch_size)
    #
    # path = "../src/saved_model/"
    # filename = "ralated_text_separate_model.h5"
    # model.save(path + filename)
    # model = tf.keras.models.load_model(path + filename)
    #
    # y_predict = list()
    # y_pred = model.predict(x_test)
    # for y in y_pred:
    #     if y[0] > y[1]:
    #         y_predict.append(0)
    #     else:
    #         y_predict.append(1)
    # print(classification_report(y_test, y_predict))
    main_url = Pretreatment.get_main_url("https://blog.csdn.net/Louis210/article/details/119666026")
    urls = Pretreatment.get_urls(main_url, verbose=False)
    print(SeparateRelatedArticle.get_course_related_urls(urls))
