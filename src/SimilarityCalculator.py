import codecs
import collections
import re

import jieba
import gensim
import lda
import numpy as np
from gensim import corpora


class SimilarityCalculator:
    """
    计算文档主题的一个工具类
    """

    @staticmethod
    def write_clean_corpus_to_file(input_filenames, output_filename):
        stopwords_file = open("./src/text/stopwords.txt")
        stopwords_string = stopwords_file.read()
        stopwords_file.close()
        my_stopwords = stopwords_string.split("\n")
        out_f = open("./src/text/" + output_filename, 'w')
        clean_txt = set()
        for input_filename in input_filenames:
            f = open("./src/text/" + input_filename, 'r')
            doc_complete = f.read()
            f.close()
            txt = jieba.lcut(doc_complete)
            for word in txt:
                word = word.strip()
                if (word in my_stopwords) or re.match("\\s", word):
                    continue
                clean_txt.add(word)
            for word in clean_txt:
                out_f.write(word + " ")
            out_f.write("\n")
        out_f.close()

    @staticmethod
    def ge_lda_model(corpus_filename):
        wordset = set()
        for eachline1 in codecs.open("./src/text/"+corpus_filename, 'r', 'utf-8'):
            lineList1 = eachline1.split(' ')
            for i in range(len(lineList1)):
                wordset.add(lineList1[i].strip())
        wordlist = list(wordset)
        wordmatrix = []
        for eachline2 in codecs.open("./src/text/"+corpus_filename, 'r', 'utf-8'):
            docwords = eachline2.strip().split(' ')
            dictionary = collections.Counter(docwords)
            key = list(dictionary.keys())
            r1 = []
            for i in range(len(wordlist)):
                if wordlist[i] in key:
                    r1.append(dictionary[wordlist[i]])
                else:
                    r1.append(0)
            wordmatrix.append(r1)
        X = np.array(wordmatrix)  # 词频矩阵
        # 模型训练
        model = lda.LDA(n_topics=20, n_iter=100, random_state=1)
        model.fit(X)

        topic_word = model.topic_word_
        # 每个topic内权重最高的5个词语
        n = 5
        print('==================topic top' + str(n) + ' word==================')
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(wordlist)[np.argsort(topic_dist)][:-(n + 1):-1]
            print('*Topic {}\n-{}'.format(i, ' '.join(topic_words)))

        doc_topic = model.doc_topic_
        # 每篇文本最可能的topic
        print('==================doc best topic==================')
        for i in range(2):
            topic_most_pr = doc_topic[i].argmax()
            print('doc: {} ,best topic: {}'.format(i, topic_most_pr))
        return model
