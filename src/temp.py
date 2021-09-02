import random
import re
import threading
import time
from multiprocessing import Process
from pprint import pprint

import requests
from baiduspider import BaiduSpider

from src.InfoReadAndWrite import InfoReadAndWrite
from src.Pretreatment import Pretreatment
from src.SimilarityFromBERT import SimilarityFromBERT
from src.SimilarityFromPMD import SimilarityFromPMD

url1 = "https://blog.csdn.net/Louis210/article/details/119666026?spm=1001.2014.3001.5501"
url2 = "https://blog.csdn.net/Prototypen/article/details/116460054"
url3 = "https://blog.csdn.net/eliforsharon/article/details/76153759"
url4 = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
#
# p1 = threading.Thread(target=get_similarities_and_write, args=(url1, 0))
# p2 = threading.Thread(target=get_similarities_and_write, args=(url2, 1))
# p1.start()
# p2.start()
urls = [url2, url1, url3]

# url = "http://icanhazip.com/ "
# print(Pretreatment.get_raw_html(url))
# def a(num):
#     print(num)
#     time.sleep(random.randrange(5,30))
#     print("{}结束".format(num))
#
#
# def n_threads_run(urls, start=0, end=-1, numbers=None):
#     """
#     开启n个线程，跑第urls[start],urls[start+1),...,urls[end-1],urls[numbers[0]],urls[numbers[1]),...,urls[numbers[m-1]]
#     :param urls: urls列表
#     :param start: 开始标号
#     :param end: 结束标号
#     :param numbers: 序号列表
#     :return: 无
#     """
#     if numbers is None:
#         numbers = []
#     threads = list()
#     for i in range(start, end):
#         thread = threading.Thread(target=a, args=(i,))
#         threads.append(thread)
#     for number in numbers:
#         thread = threading.Thread(target=a, args=(number,))
#         threads.append(thread)
#     for thread in threads:
#         time.sleep(1)
#         thread.start()
#
# n_threads_run(urls, numbers=[0, 1, 2, 3])
# while True:
#     length = len(threading.enumerate())
#     print('当前运行的线程数为：%d' % length)
#     if length <= 3:
#         n_threads_run(urls, numbers=[3, 4])
#     time.sleep(0.5)

# def a(num):
#     print(num)
#     time.sleep(20)
#     print("{}结束".format(num))
#
#
# def n_threads_run(urls, number_list, num_worker=20):
#     while True:
#         if len(threading.enumerate()) < num_worker + 1:
#             i = min(number_list)
#             number_list.remove(i)
#             thread = threading.Thread(target=a, args=(i,))
#             thread.start()
#         time.sleep(1)
#
#
# n_threads_run([], list(range(10, 20)) + [1, 3, 5], num_worker=5)
#
#
# @staticmethod
# def n_threads_run(urls, start=0, end=-1, numbers=None):
#     """
#     开启n个线程，跑第urls[start],urls[start+1),...,urls[end-1],urls[numbers[0]],urls[numbers[1]),...,urls[numbers[m-1]]
#     :param urls: urls列表
#     :param start: 开始标号
#     :param end: 结束标号
#     :param numbers: 序号列表
#     :return: 无
#     """
#     if numbers is None:
#         numbers = []
#     threads = list()
#     for i in range(start, end):
#         thread = threading.Thread(target=InfoReadAndWrite.get_similarities_and_write, args=(urls[i], i))
#         threads.append(thread)
#     for number in numbers:
#         thread = threading.Thread(target=InfoReadAndWrite.get_similarities_and_write, args=(urls[number], number))
#         threads.append(thread)
#     for thread in threads:
#         time.sleep(random.randrange(5, 10, 1))
#         thread.start()
text = "让我看看1\n/*\n111111\n*/让我看看2\n'''\n22222\n'''让我看看3\n123#333\n456//444\n//666\n9 9 0"
print(text)
print("-----------------")
print(Pretreatment.clean_code_for_text(text))
