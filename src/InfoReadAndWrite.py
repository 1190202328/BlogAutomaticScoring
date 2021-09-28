import os
import random
import re
import threading
import time
from pprint import pprint
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

from src import GetWebResource
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
    def get_urls(filepath):
        """
        从磁盘中读如urls
        :param filepath: 路径
        :return: urls列表
        """
        urls = list()
        # tags = list()
        f = open(filepath, "r")
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
            results.append(GetWebResource.split_txt(url, EDU=False))
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
    def write_similarity_to_file(similarity, file_path, paragraph_len=80, sentence_len=80, code_len=60):
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
        if os.path.exists(file_path):
            print("\n" + file_path + ">>>已存在\n", end="")
        else:
            with open(file_path, mode='w') as f:
                writer = csv.writer(f)
                writer.writerows([results])

    @staticmethod
    def get_similarities(filepath):
        """
        获取similarities
        :return: similarities
        """
        similarities = list()
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                similarities.append(row)
        return np.array(similarities).astype(np.float)

    @staticmethod
    def get_similarities_and_write(url, num, data_filepath, verbose=True, pre_verbose=True):
        """
        获取similarity并且将其格式化地写入文件，文件名称为similarities_num.csv
        :param data_filepath: 文件名称
        :param url: url地址
        :param num: 第多少个url
        :return: 无
        """
        similarity = SimilarityFromBERT.get_5d_similarities(url, EDU=False, verbose=verbose, pre_verbose=pre_verbose)
        if not similarity:
            print("\n到此url停止>>>{}\n".format(url), end="")
            return 0
        InfoReadAndWrite.write_similarity_to_file(similarity, data_filepath[:len(data_filepath)-4]+'{}.csv'.format(num))

    @staticmethod
    def merge_to_main_csv(start, end, data_filepath):
        """
        例子：data_filepath = 'similarities.csv'
        将data_filepath[:len(data_filepath)-4]{start}.csv,data_filepath[:len(data_filepath)-4]{start+1}.csv,
        ...,data_filepath[:len(data_filepath)-4]{end-1}.csv 合并到 data_filepath
        :param start: 开始序号
        :param end: 结束序号
        :return: 无
        """
        similarities = list()
        for i in range(start, end):
            with open(data_filepath[:len(data_filepath)-4]+'{}.csv'.format(i), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    similarities.append(row)
        with open(data_filepath, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(similarities)

    @staticmethod
    def n_threads_run(urls, number_list, data_filepath, num_worker=20):
        """
        保持总有num_worker个线程在运行
        :param data_filepath: 文件名
        :param urls: url列表
        :param number_list: 序号列表
        :param num_worker: 线程数
        :return: 无
        """
        while True:
            if len(threading.enumerate()) < num_worker + 1:
                if not number_list:
                    break
                i = min(number_list)
                number_list.remove(i)
                thread = threading.Thread(target=InfoReadAndWrite.get_similarities_and_write, args=(urls[i], i, data_filepath))
                thread.start()
            time.sleep(60)

    @staticmethod
    def get_number_list(template_filepath, min_num, max_num):
        """
        获取还没有取爬虫获取的序号列表
        :param template_filepath: 模板路径，例如:../src/text/similarities_bigger
        :param max_num: 最大的序号
        :param min_num: 最小的序号
        :return: 还没有取爬虫获取的序号列表
        """
        number_list = []
        for i in range(min_num, max_num):
            if not os.path.exists(template_filepath + i.__str__() + ".csv"):
                number_list.append(i)
        return number_list

    @staticmethod
    def write_to_csv(filepath_read, filepath_write):
        """
        将以\t为分隔符的url.txt文件转换为.csv文件
        :param filepath_write: .csv文件
        :param filepath_read:  \t为分隔符的url.txt文件
        :return: 无
        """
        urls = list()
        tags = list()
        with open(filepath_read, "r") as f:
            for line in f.readlines():
                line = re.sub('\t+', '\t', line)
                r = line.split("\t")
                urls.append(r[1])
                if len(r) > 2:
                    tags.append(r[2].replace('\n', ''))
        workbook = xlsxwriter.Workbook(filepath_write)
        worksheet = workbook.add_worksheet("sheet1")
        for i in range(len(urls)):
            worksheet.write(i, 0, urls[i])
            if i < len(tags):
                worksheet.write(i, 1, tags[i])
        workbook.close()


if __name__ == '__main__':
    filepath = "../src/text/按原创性分类.txt"
    data_filepath = "../src/text/similarities_bigger_new.csv"
    urls = InfoReadAndWrite.get_urls(filepath)
    print("shape=", end="")
    print(InfoReadAndWrite.get_similarities(data_filepath).shape)
    number_list = InfoReadAndWrite.get_number_list(data_filepath[:len(data_filepath)-4], 0, 0)
    print("[{}]>>>".format(len(number_list)), end="")
    print("未完成爬虫的序号>>>", end="")
    print(number_list)
    # InfoReadAndWrite.n_threads_run(urls, number_list, data_filepath, num_worker=30)

    # InfoReadAndWrite.merge_to_main_csv(0, 0, data_filepath)
    # print(InfoReadAndWrite.get_similarities(data_filepath).shape)