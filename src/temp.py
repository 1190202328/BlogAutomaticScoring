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
urls = [url2, url1, url3]

# url = "http://icanhazip.com/ "
# print(Pretreatment.get_raw_html(url))

# text = "让我看看1\n/*\n111111\n*/让我看看2\n'''\n22222\n'''让我看看3\n123#333\n456//444\n//666\n9 9 0"
# print(text)
# print("-----------------")
# print(Pretreatment.clean_code_for_text(text))

for i in range(300):
    texts, _ = Pretreatment.get_related_head('java', 1, verbose=False)
    print('{}>>>'.format(i), end='')
    print(len(texts))

