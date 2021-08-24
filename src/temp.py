import threading
import time
from multiprocessing import Process
from pprint import pprint

import requests

from src.InfoReadAndWrite import InfoReadAndWrite
from src.Pretreatment import Pretreatment
from src.SimilarityFromBERT import SimilarityFromBERT
from src.SimilarityFromPMD import SimilarityFromPMD

# code = "public static void main(String[] args) {"
# # codes = Pretreatment.get_related_codes(code, 10)
# # for c in codes:
# #     print(">>>"+c)
# code1 = "return weight;"
# code2 = "return weight"
# code3 = "private static void main(String[] args) {"
# print(SimilarityFromPMD.is_similar(code, code3))
# code_similarity = SimilarityFromBERT.get_similarity([code1, code3])[:1][0][1]
# print(code_similarity)
# code4 = "================================================================================ */"
# print(SimilarityFromPMD.is_similar(code1, code4))
# code_similarity = SimilarityFromBERT.get_similarity([code1, code4])[:1][0][1]
#
# print(code_similarity)

url1 = "https://blog.csdn.net/Louis210/article/details/119666026?spm=1001.2014.3001.5501"
url2 = "https://blog.csdn.net/Prototypen/article/details/116460054"
url3 = "https://blog.csdn.net/eliforsharon/article/details/76153759"

#
# p1 = threading.Thread(target=get_similarities_and_write, args=(url1, 0))
# p2 = threading.Thread(target=get_similarities_and_write, args=(url2, 1))
# p1.start()
# p2.start()
urls = ["", url2, url1, url3]
# InfoReadAndWrite.n_threads_run_interval(urls, [1, 3])

# headers = {
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#             'Accept-Encoding': 'gzip, deflate, compress',
#             'Accept-Language': 'en-us;q=0.5,en;q=0.3',
#             'Cache-Control': 'max-age=0',
#             'Connection': 'keep-alive',
#             'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
#         }
# # proxy = "127.0.0.1:11000"
# proxy = "127.0.0.1:7890"  # clashX
# proxies = {
#     'http': 'http://' + proxy,
#     'https': 'https://' + proxy
# }
# print('http://' + proxy)
# r = requests.get(url=url1, headers=headers, timeout=10, proxies=proxies)
# print(r.status_code)
# r.raise_for_status()
