import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.InfoReader import InfoReader
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
        texts = Pretreatment.clean_with_low_frequency(texts)
        if not vocab_list:
            vocab_list = list()
            f = open("../src/text/vocab_list_text.txt", 'r')
            for line in f.readlines():
                vocab_list.append(line[:-1])
            f.close()
        token_list = list()
        for text in texts:
            token = list()
            for word in text:
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
        return [],[]


if __name__ == '__main__':
    related_urls = SeparateRelatedArticle.get_urls("../src/text/urls.txt")
    # print(len(related_urls))
    # for url in related_urls:
    #     print(url)
    total_urls = list()
    students = InfoReader.get_student_info("../src/text/学生个人博客信息.xlsx")
    for student in students:
        if student.url is not None:
            main_url = Pretreatment.get_main_url(student.url)
            if main_url:
                urls = Pretreatment.get_urls(main_url)
                if not urls:
                    print(">>>"+main_url)
                total_urls += urls
    total_urls = set(total_urls).difference(related_urls)
    with open("../src/text/非软件构造urls.txt", mode="w") as f:
        for url in total_urls:
            f.write(url)
            f.write("\n")




    texts, labels = SeparateRelatedArticle.get_texts_and_labels()
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=0)
    vocab = set()
    for x in x_train:
        for word in x:
            vocab.add(word)

    # 深度学习分类
    vocab_list = list()
    vocab_list.append("<paddle>")
    vocab_list.append("<unk>")
    vocab_list += list(sorted(vocab))

    f = open("../src/text/vocab_list_text.txt", 'w')
    for vocab in vocab_list:
        f.write(vocab)
        f.write("\n")
    f.close()

    token_list = []
    embedding_len = 200
    output_dim = 64
    learning_rate = 0.1
    batch_size = 330
    epochs = 1
    verbose = 2
    vocab_len = len(vocab_list)
    print("词典大小:{}".format(vocab_len))

    x_train = SeparateRelatedArticle.get_sequences(x_train, embedding_len, vocab_list)
    x_test = SeparateRelatedArticle.get_sequences(x_test, embedding_len, vocab_list)

    input_token = tf.keras.Input(shape=(embedding_len,))
    embedding = tf.keras.layers.Embedding(input_dim=vocab_len, output_dim=output_dim)(input_token)
    embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim))(embedding)
    output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding)
    model = tf.keras.Model(input_token, output)
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    print(model.summary())
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, validation_freq=2, verbose=verbose,
              batch_size=batch_size)

    path = "../src/saved_model/"
    filename = "ralated_text_separate_model.h5"
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
