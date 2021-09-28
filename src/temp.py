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

from src import SearchWeb, Global
from src.Crawl import Crawl
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

# def a(num):
#     notice = tqdm(total=num, position=True)
#     notice.set_description("[{}] ".format(num))
#     for i in range(num):
#         time.sleep(0.1)
#         notice.update()
# threading.Thread(target=a, args=(100,)).start()
# threading.Thread(target=a, args=(200,)).start()

# result = Pretreatment.split_txt("https://blog.csdn.net/kingofthecool/article/details/107290170")
# result = Pretreatment.split_txt(url1)
# pprint(result)
# # SimilarityFromBERT.get_5d_similarities("https://blog.csdn.net/kingofthecool/article/details/107290170", verbose=True)
# Pretreatment.get_related_head("https://blog.csdn.net/kingofthecool/article/details/107290170", 1)
# Pretreatment.get_related_head('https://blog.csdn.net/weixin_43872188/article/details/106373291', 1)
# pprint(BaiduSpider().search_web('软件构造复习（一）软件构造的多维视图和软件构造的质量标准', pn=1, exclude=['all']).get('results'))
# pprint(BaiduSpider().search_web('日志管理库log4j和log4j2的安装配置指南', pn=1, exclude=['all']).get('results'))
# print(SearchWeb.get_related_head_and_text(result['head'], result['date']))

'http://baidu.com/s?wd=## 二、知识复习&rn=50&oq=## 二、知识复习&ie=utf-8'

# print(Crawl.get_raw_html('', True))

# s = '## 二、   知识复习'
s = '%p:表示日志记录的异常的类别或者说日志的级别；'
s = Pretreatment.clean_to_search_keywords(s)
print(s)
baidu_url = 'http://baidu.com/s?wd=' + s + "&rn=50" + "&oq=" + s + "&ie=utf-8"
# s = '复习了软件的多维视图'
# s = 'http://baidu.com/s?wd=复习了软件的多维视图&rn=50&oq=复习了软件的多维视图&ie=utf-8'
print(baidu_url)
bf = BeautifulSoup(Pretreatment.get_raw_html(baidu_url), "html.parser")
content = bf.get_text()
pprint(content)
# print(re.match(Global.not_find, content, flags=re.S))

# print(Pretreatment.get_related_paragraphs_and_sentences('复习了软件的多维视图'))