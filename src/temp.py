import random
import re
import threading
import time
from multiprocessing import Process
from pprint import pprint

import jieba
import requests
from baiduspider import BaiduSpider
from bs4 import BeautifulSoup
from tqdm import tqdm

from src import SearchWeb, Global, HTML
from src.InfoReadAndWrite import InfoReadAndWrite
from src.SimilarityFromBERT import SimilarityFromBERT
from src.SimilarityFromPMD import SimilarityFromPMD

url1 = "https://blog.csdn.net/Louis210/article/details/119666026?spm=1001.2014.3001.5501"
url2 = "https://blog.csdn.net/Prototypen/article/details/116460054"
url3 = "https://blog.csdn.net/eliforsharon/article/details/76153759"
url4 = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
urls = [url2, url1, url3]

# url = "http://icanhazip.com/ "
# print(HTML.get_raw_html_origin(url))

# def a(num):
#     notice = tqdm(total=num, position=True)
#     notice.set_description("[{}] ".format(num))
#     for i in range(num):
#         time.sleep(0.1)
#         notice.update()
# threading.Thread(target=a, args=(100,)).start()
# threading.Thread(target=a, args=(200,)).start()
