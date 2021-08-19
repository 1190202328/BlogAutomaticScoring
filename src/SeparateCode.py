import re
from pprint import pprint

import jieba
import numpy as np
import tensorflow as tf


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
    def get_sequences(sentences, embedding_len, vocab_list=""):
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
            for line in f.readlines():
                vocab_list.append(line[:-1])
            f.close()
        token_list = list()
        for sentence in sentences:
            token = list()
            for word in sentence:
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
        sentences = text.split("\n")
        code_like_sentences_list = list()
        code_like_sentences = list()
        for sentence in sentences:
            pre_sentence = sentence.lower().strip()
            pre_sentence = SeparateCode.clean_code_line(pre_sentence, low_limit=0, high_limit=100)
            if pre_sentence == "" or re.match(r'[\u4e00-\u9fa5]+', pre_sentence):
                continue
            if re.match(r'.*[\u4e00-\u9fa5]+.*', pre_sentence):
                if not re.match(r'([^\u4e00-\u9fa5]*".*?"[^\u4e00-\u9fa5]*)+', sentence):
                    continue
            code_like_sentences.append(sentence)
            pre_sentence = re.split("[ .]", pre_sentence)
            # pre_sentence = jieba.lcut(pre_sentence)
            code_like_sentences_list.append(pre_sentence)
        # pprint(sentences)
        if not code_like_sentences:
            return codes
        print("可能是代码的如下:")
        pprint(code_like_sentences)
        code_indexes = [1] * len(code_like_sentences_list)
        sequences = SeparateCode.get_sequences(code_like_sentences_list, embedding_len)
        # print(sequences)
        path = "../src/saved_model/"
        filename = "code_separate_model.h5"
        model = tf.keras.models.load_model(path + filename)
        results = model.predict(sequences)
        for i in range(len(results)):
            if results[i][0] > results[i][1]:
                code_indexes[i] = 0
        # pprint(code_indexes)
        for i in range(1, len(code_indexes) - 1):
            if code_indexes[i] == 1 and code_indexes[i - 1] == 0 and code_indexes[i + 1] == 0:
                code_indexes[i] = 0
        for i in range(len(code_indexes)):
            if code_indexes[i] == 0:
                codes.append(code_like_sentences[i])
        print("检测出的代码的如下:")
        pprint(codes)
        return codes


if __name__ == '__main__':
    # url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    # text = Pretreatment.split_txt(url).get('hole_text')
    # text = "public static void main\nI love you"

    f = open("../src/text/江江.txt")
    text = f.read()
    f.close()
    pprint(SeparateCode.get_codes(text))
