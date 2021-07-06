import codecs
import collections
from gensim import corpora
import pprint
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
    def get_documents(document_filenames):
        """
        从文件中读取文本到内存中
        :param document_filenames: 文件名称，如："软件构造.txt"。该文档应该存储在/src/text/目录下。
        :return: 文件列表，每个元素为一个文档
        """
        documents = list()
        for document_filename in document_filenames:
            f = open("./src/text/" + document_filename, 'r')
            documents.append(f.read())
            f.close()
        return documents

    @staticmethod
    def clean(documents):
        """
        将按列表存储的文档进行清洗
        :param documents: 按列表存储的文档，列表中一个元素为一个文档
        :return: 清洗好的文档，二维列表，一行为一个文档的清洗后的词
        """
        stopwords_file = open("./src/text/stopwords.txt")
        stopwords_string = stopwords_file.read()
        stopwords_file.close()
        my_stopwords = stopwords_string.split("\n")
        texts = list()
        for document in documents:
            text = list()
            for word in jieba.cut(document):
                word = word.strip()
                word = word.lower()
                if (word in my_stopwords) or re.match("\\s", word):
                    continue
                text.append(word)
            texts.append(text)
        # 删除频率<=1的词
        frequency = collections.defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]
        return texts

    @staticmethod
    def save_dictionary(texts, dictionary_name):
        """
        清洗好的文档（二维列表，一行为一个文档的清洗后的词）添加到字典中并以(dictionary_name.dict)为名字保存在/src/text下
        :param texts: 清洗好的文档
        :param dictionary_name: 字典名
        :return: 无
        """
        dictionary = corpora.Dictionary(texts)
        dictionary.save("./src/text/"+dictionary_name+".dict")  # store the dictionary, for future reference

    @staticmethod
    def get_dictionary(dictionary_name):
        """
        获得存储好的词典
        :param dictionary_name: 词典名称，如："测试词典"（不含后缀）
        :return: 该词典
        """
        return corpora.Dictionary.load("./src/text/"+dictionary_name+".dict")

    @staticmethod
    def save_corpus(texts, dictionary, corpus_name):
        """
        将词典转换为词袋向量，并以(corpus_name.mm)为名字保存在/src/text下
        :param texts: 清洗好的文档（二维列表，一行为一个文档的清洗后的词）
        :param dictionary: 词典
        :param corpus_name: 词袋向量
        :return: 无
        """
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize("./src/text/"+corpus_name+".mm", corpus)  # store to disk, for later use

    @staticmethod
    def get_corpus(corpus_name):
        """
        获得存储好的词袋向量
        :param corpus_name: 词袋向量文件名字，例如："测试词袋向量"（不含后缀）
        :return: 词袋向量
        """
        return corpora.MmCorpus("./src/text/"+corpus_name+".mm")

    @staticmethod
    def get_tfidf_model(corpus_filename):
        # text_corpus = set()
        # for eachline in codecs.open("./src/text/"+corpus_filename, 'r', 'utf-8'):
        #     text_corpus.add(eachline)
        # texts = [[word for word in document.split(" ")]for document in text_corpus]
        text_corpus = [
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time",
            "The EPS user interface management system",
            "System and human system engineering testing of EPS",
            "Relation of user perceived response time to error measurement",
            "The generation of random binary unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors IV Widths of trees and well quasi ordering",
            "Graph minors A survey",
        ]
        import pprint
        # Create a set of frequent words
        stoplist = set('for a of the and to in'.split(' '))
        # Lowercase each document, split it by white space and filter out stopwords
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in text_corpus]
        print("texts如下：")
        print(texts)
        print("texts如上：")
        # Count word frequencies
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        # Only keep words that appear more than once
        processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
        pprint.pprint(processed_corpus)
        from gensim import corpora
        dictionary = corpora.Dictionary(processed_corpus)
        print(dictionary)
        pprint.pprint(dictionary.token2id)  # 打印词典内容
        bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
        from gensim import models
        # train the model
        tfidf = models.TfidfModel(bow_corpus)
        from gensim import similarities
        index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)
        query_document = 'system engineering'.split()
        query_bow = dictionary.doc2bow(query_document)
        sims = index[tfidf[query_bow]]
        for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
            print(document_number, score)

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
