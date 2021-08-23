import threading
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
        # tags = list()
        f = open(path + filename, "r")
        for line in f.readlines():
            r = line.split("\t")
            urls.append(r[1])
            # tags.append(r[3])
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
        """
        获得根据urls得到的results
        :return: results
        """
        with open('../src/text/results.txt', 'r') as f:
            return json.loads(f.read())

    @staticmethod
    def show_results_data():
        """
        分析results
        :return: 无
        """
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
    def write_similarity_to_file(similarity, file_path, paragraph_len=50, sentence_len=70, code_len=10):
        """
        将similarity格式化地写入到文件中
        :param similarity: 相似度列表
        :param file_path: 文件路径
        :param paragraph_len: 标准段落数
        :param sentence_len: 标准句子数
        :param code_len: 标准代码数
        :return: 无
        """
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
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([results])

    @staticmethod
    def get_similarities():
        """
        获取similarities
        :return: similarities
        """
        similarities = list()
        with open('../src/text/similarities.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                similarities.append(row)
        return np.array(similarities).astype(np.float)

    @staticmethod
    def get_similarities_and_write(url, num):
        """
        获取similarit并且将其格式化地写入文件，文件名称为similarities_num.csv
        :param url: url地址
        :param num: 第多少个url
        :return: 无
        """
        print("url>>>" + url)
        similarity = SimilarityFromBERT.get_5d_similarities(url)
        if not similarity:
            print("到此url停止>>>" + url)
            return 0
        InfoReadAndWrite.write_similarity_to_file(similarity, '../src/text/similarities_{}.csv'.format(num))

    @staticmethod
    def n_threads_run(urls, start, end):
        """
        开启n个线程，一次性跑end-start篇文章(跑第urls[start],urls[start+1),...,urls[end-1]
        :param urls: urls列表
        :param start: 开始标号
        :param end: 结束标号
        :return: 无
        """
        threads = list()
        for i in range(start, end):
            thread = threading.Thread(target=InfoReadAndWrite.get_similarities_and_write, args=(urls[i], i))
            threads.append(thread)
        for thread in threads:
            thread.start()

    @staticmethod
    def merge_to_main_csv(start, end):
        """
        将similarities_{start}.csv,similarities_{start+1}.csv,...,similarities_{end-1}.csv 合并到 ../src/text/similarities.csv
        :param start: 开始序号
        :param end: 结束序号
        :return: 无
        """
        similarities = list()
        for i in range(start, end):
            with open('../src/text/similarities_{}.csv'.format(i), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    similarities.append(row)
        with open('../src/text/similarities.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(similarities)

    @staticmethod
    def n_threads_run_interval(urls, numbers):
        """
        开启n个线程，一次性跑len(numbers)篇文章(跑第urls[numbers[0]],urls[numbers[1]),...,urls[numbers[n-1]]
        :param urls: urls列表
        :param numbers: 序号列表
        :return: 无
        """
        threads = list()
        for number in numbers:
            thread = threading.Thread(target=InfoReadAndWrite.get_similarities_and_write, args=(urls[number], number))
            threads.append(thread)
        for thread in threads:
            thread.start()


if __name__ == '__main__':
    path = "../src/text/"
    filename = "urls.txt"
    urls = InfoReadAndWrite.get_urls(path, filename)
    # InfoReadAndWrite.write_result(urls)

    # InfoReadAndWrite.show_results_data()

    # i = 0
    # pprint(urls)
    # for url in urls:
    #     with open('../src/text/urls.txt', 'a') as f:
    #         f.write(i.__str__() + "\t" + url + "\n")
    #         i += 1

    # print(InfoReadAndWrite.get_similarities().shape)
    # InfoReadAndWrite.n_threads_run(urls, 33, 35)

    # InfoReadAndWrite.n_threads_run_interval(urls, [])
    # InfoReadAndWrite.merge_to_main_csv(1, 5)
