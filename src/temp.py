import threading
import time
from multiprocessing import Process
from pprint import pprint

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

from tqdm import tqdm
import time
def a(num):
    pbar = tqdm(total=num)
    pbar.set_description("{}".format(num))
    for i in range(num):
        pbar.update(1)
        time.sleep(0.01)
    pbar.close()


p1 = threading.Thread(target=a, args=(500,))
p2 = threading.Thread(target=a, args=(200,))
p1.start()
p2.start()
