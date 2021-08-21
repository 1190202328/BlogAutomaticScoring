from pprint import pprint
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from src.Pretreatment import Pretreatment
from src.SimilarityFromBERT import SimilarityFromBERT


def padding(result, padding_len, padding_content):
    r = list(result)
    r = r[:padding_len] + [padding_content] * (padding_len - len(r))
    return r


class InfoReadAndWrite:
    """
    从爬虫上读取的信息读写工具类
    """

    @staticmethod
    def get_urls(path, filename):
        """
        从磁盘中读如urls
        :param path: 路径
        :param filename: 文件名
        :return: urls列表
        """
        urls = list()
        f = open(path + filename, "r")
        for line in f.readlines():
            urls.append(line[:-1])
        f.close()
        return urls

    @staticmethod
    def write_result(urls):
        """
        获取urls的result，并将result列表写入磁盘
        :param urls: url列表
        :return: 无
        """
        results = list()
        i = 1
        for url in urls:
            print("正在处理第{}个url(共{}个)>>>".format(i, len(urls)) + url)
            i += 1
            results.append(Pretreatment.split_txt(url, EDU=False))
        with open('../src/text/results.txt', 'w') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=2))

    @staticmethod
    def get_results():
        with open('../src/text/results.txt', 'r') as f:
            return json.loads(f.read())

    @staticmethod
    def show_results_data():
        results = InfoReadAndWrite.get_results()
        print("共{}个可能的结果".format(len(results)))
        valid_results_len = 0
        paragraphs_len = list()
        sentences_len = list()
        codes_len = list()
        for result in results:
            if result:
                valid_results_len += 1
                if result['paragraphs']:
                    paragraphs_len.append(len(result['paragraphs']))
                if result['sentences']:
                    sentences_len.append(len(result['sentences']))
                if result['codes']:
                    codes_len.append(len(result['codes']))
        paragraphs_len = sorted(paragraphs_len)
        sentences_len = sorted(sentences_len)
        codes_len = sorted(codes_len)
        print("共{}个有效结果".format(valid_results_len))
        print("共{}个段落".format(len(paragraphs_len)))
        print("最大个数:{};最小个数:{};平均个数:{};中位数:{}".format(max(paragraphs_len), min(paragraphs_len), np.mean(paragraphs_len),
                                                      np.median(paragraphs_len)))
        print("共{}个句子".format(len(sentences_len)))
        print("最大个数:{};最小个数:{};平均个数:{};中位数:{}".format(max(sentences_len), min(sentences_len), np.mean(sentences_len),
                                                      np.median(sentences_len)))
        print("共{}个代码".format(len(codes_len)))
        print("最大个数:{};最小个数:{};平均个数:{};中位数:{}".format(max(codes_len), min(codes_len), np.mean(codes_len),
                                                      np.median(codes_len)))
        x1 = [i for i in range(len(paragraphs_len))]
        my_x_ticks = np.arange(0, 500, 20)
        plt.yticks(my_x_ticks)
        plt.plot(x1, paragraphs_len)
        plt.show()
        x2 = [i for i in range(len(sentences_len))]
        my_x_ticks = np.arange(0, 700, 20)
        plt.yticks(my_x_ticks)
        plt.plot(x2, sentences_len)
        plt.show()
        x3 = [i for i in range(len(codes_len))]
        my_x_ticks = np.arange(0, 80, 5)
        plt.yticks(my_x_ticks)
        plt.plot(x3, codes_len)
        plt.show()

    @staticmethod
    def write_similarity_to_file(similarity, paragraph_len=50, sentence_len=70, code_len=10):
        heads = similarity['head']
        texts = similarity['text']
        paragraphs = similarity['paragraph']
        sentences = similarity['sentence']
        codes = similarity['code']
        results = list()
        results += heads
        results += texts
        paragraphs = padding(paragraphs, paragraph_len, [1] * 10)
        sentences = padding(sentences, sentence_len, [1] * 10)
        codes = padding(codes, code_len, [1] * 10)
        for paragraph in paragraphs:
            results += paragraph
        for sentence in sentences:
            results += sentence
        for code in codes:
            results += code
        with open('../src/text/similarities.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows([results])

    @staticmethod
    def get_similarities():
        similarities = list()
        with open('../src/text/similarities.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                similarities.append(row)
        return np.array(similarities).astype(np.float)


if __name__ == '__main__':
    path = "../src/text/"
    filename = "url总集.txt"
    urls = InfoReadAndWrite.get_urls(path, filename)
    # InfoReadAndWrite.write_result(urls)

    # InfoReadAndWrite.show_results_data()

    for url in urls[:5]:
        similarity = SimilarityFromBERT.get_5d_similarities(url)
        InfoReadAndWrite.write_similarity_to_file(similarity)
    print(InfoReadAndWrite.get_similarities().shape)
