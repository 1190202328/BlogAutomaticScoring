import collections
import re

import gensim
import jieba
from gensim import corpora


class SimilarityCalculator:
    """
    计算文档主题的一个工具类
    """

    @staticmethod
    def get_documents(path, document_filenames):
        """
        从文件中读取文本到内存中
        :param path: 路径名，如："./src/text/"
        :param document_filenames: 文件名称，如："软件构造.txt"
        :return: 文件列表，每个元素为一个文档
        """
        documents = list()
        for document_filename in document_filenames:
            f = open(path + document_filename, 'r')
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
    def save_dictionary(texts, path, dictionary_name):
        """
        清洗好的文档（二维列表，一行为一个文档的清洗后的词）添加到字典中并以(dictionary_name.dict)为名字保存在/src/text下
        :param path: 路径名，如："./src/text/"
        :param texts: 清洗好的文档
        :param dictionary_name: 字典名
        :return: 无
        """
        dictionary = corpora.Dictionary(texts)
        dictionary.save(path + dictionary_name + ".dict")  # store the dictionary, for future reference

    @staticmethod
    def generate_dictionary(texts):
        """
        将清洗好的文档（二维列表，一行为一个文档的清洗后的词）添加到字典中
        :param texts: 清洗好的文档
        :return: 字典
        """
        return corpora.Dictionary(texts)

    @staticmethod
    def get_dictionary(path, dictionary_name):
        """
        获得存储好的词典
        :param path: 路径名，如："./src/text/"
        :param dictionary_name: 词典名称，如："测试词典"（不含后缀）
        :return: 该词典
        """
        return corpora.Dictionary.load(path + dictionary_name + ".dict")

    @staticmethod
    def save_corpus(texts, dictionary, path, corpus_name):
        """
        将词典转换为词袋向量，并以(corpus_name.mm)为名字保存在/src/text下
        :param path: 路径名，如："./src/text/"
        :param texts: 清洗好的文档（二维列表，一行为一个文档的清洗后的词）
        :param dictionary: 词典
        :param corpus_name: 词袋向量
        :return: 无
        """
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(path + corpus_name + ".mm", corpus)  # store to disk, for later use

    @staticmethod
    def get_corpus(path, corpus_name):
        """
        获得存储好的词袋向量
        :param path: 路径名，如："./src/text/"
        :param corpus_name: 词袋向量文件名字，例如："测试词袋向量"（不含后缀）
        :return: 词袋向量
        """
        return corpora.MmCorpus(path + corpus_name + ".mm")

    @staticmethod
    def save_lsi_index(corpus, dictionary, num_topics, path, index_name):
        """
        获得lsi的索引，并以(index_name.index)为名字保存在/src/text下
        :param path: 路径名，如："./src/text/"
        :param num_topics: 主题数量
        :param index_name: 索引名字
        :param corpus: 处理好的词袋向量
        :param dictionary: 处理好的词典
        :return: 无
        """
        lsi = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        index = gensim.similarities.MatrixSimilarity(lsi[corpus])  # 将语料转换到LSI空间并建立索引
        index.save(path + index_name + ".index")

    @staticmethod
    def get_lsi_model(corpus, dictionary, num_topics):
        """
        获得一个lsi模型
        :param dictionary: 处理好的词典
        :param corpus: 处理好的词袋向量
        :param num_topics: 主题数量
        :return:lsi模型
        """
        return gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

    @staticmethod
    def get_lsi_index(path, index_name):
        """
        获得存储好的lsi索引
        :param path: 路径名，如："./src/text/"
        :param index_name: 词袋向量文件名字，例如："测试索引"（不含后缀）
        :return: 索引
        """
        return gensim.similarities.MatrixSimilarity.load(path + index_name + ".index")

    @staticmethod
    def train_lsi_model(path, document_filenames, model_related_filename):
        """
        一键训练模型，并且将模型相关数据以model_related_filename保存在path路径下
        :param path:路径名称
        :param document_filenames:需要训练的文档名称集合
        :param model_related_filename:模型相关数据的名称
        :return:无
        """
        documents = SimilarityCalculator.get_documents(path, document_filenames)
        texts = SimilarityCalculator.clean(documents)
        SimilarityCalculator.save_dictionary(texts, path, model_related_filename)
        dictionary = SimilarityCalculator.get_dictionary(path, model_related_filename)
        SimilarityCalculator.save_corpus(texts, dictionary, path, model_related_filename)
        corpus = SimilarityCalculator.get_corpus(path, model_related_filename)
        SimilarityCalculator.save_lsi_index(corpus, dictionary, len(document_filenames), path, model_related_filename)

    @staticmethod
    def get_similarity(index, document, dictionary, lsi):
        """
        得到文档的相似度
        :param lsi: lsi模型
        :param dictionary: 词典
        :param index: 索引
        :param document: 文档（字符串）
        :return: 文章与0号主题的相似度小于limit，则返回False；反之返回True
        """
        vec_bow = dictionary.doc2bow(jieba.lcut(document.lower()))
        vec_lsi = lsi[vec_bow]  # 将查询文本转换到LSI模型的向量空间
        sims = index[vec_lsi]  # 获取查询文本和语料中文本的相似度
        similarity = 0
        for i, s in enumerate(sims):
            similarity = s
            break
        # print("相似度为：{0}".format(similarity))
        return similarity
