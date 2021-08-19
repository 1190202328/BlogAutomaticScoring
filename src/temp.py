from pprint import pprint

from src.Pretreatment import Pretreatment
from src.SimilarityFromBERT import SimilarityFromBERT

# url = "https://blog.csdn.net/Louis210/article/details/117415546"
# result = Pretreatment.split_txt(url)
# pprint(SimilarityFromBERT.get_text_related(result['sentences'], result['text']))

url = "https://starlooo.github.io/2021/07/02/CaiKeng/"
result = Pretreatment.split_txt(url)
print("text如下：")
pprint(result['text'])
print("代码如下")
pprint(result['codes'])