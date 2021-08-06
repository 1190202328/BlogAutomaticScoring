import re
from pprint import pprint

from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient

from src import demo
from src.Pretreatment import Pretreatment
from src.SimilarityCalculator import SimilarityCalculator


class SimilarityFromBERT:
    """
    使用BERT来计算相似度
    """

    @staticmethod
    def get_similarity(sentences):
        """
        输入句子，获得相似度矩阵
        :param sentences: 句子列表
        :return: 相似度矩阵，维度为句子数*句子数
        """
        if len(sentences) <= 1:
            return [[0.0, 0.0], [0.0, 0.0]]
        stopwords_file = open("../src/text/stopwords.txt")
        stopwords_string = stopwords_file.read()
        stopwords_file.close()
        my_stopwords = stopwords_string.split("\n")

        bc = BertClient(check_length=False)
        clean_sentences = SimilarityCalculator.clean_with_low_frequency(sentences, my_stopwords)
        sentences = [" ".join(clean_sentence) for clean_sentence in clean_sentences]
        print(sentences)
        return cosine_similarity(bc.encode(sentences))

    @staticmethod
    def get_5d_similarities(url, head_number, text_number, paragraph_number, sentence_number, code_number=5,
                            sentence_lenth_limit=5, EDU=False):
        """
        通过url链接获取5个维度的相似度：1。标题 2。全文 3。段落 4。句子 5。代码（都进行了停用词处理）（搜索结果中不包含自己）
        :param url: url链接
        :return: 字典，head:标题对应相似度。text:全文对应相似度。paragraph:段落对应相似度。sentence:句子对应相似度。code:代码对应相似度
        """
        similarity = dict()
        result = Pretreatment.split_txt(url)
        head = result['head']
        text = result['text']
        paragraphs = result['paragraphs']
        sentences = result['sentences']
        # 标题相似度
        # related_heads = Pretreatment.get_related_head(head, head_number)
        # similarity['head'] = SimilarityFromBERT.get_similarity([head]+related_heads)[:1][0][1:]
        # print(related_heads)
        # print(len(similarity['head']))
        # print(similarity['head'])
        # 全文相似度
        # text_related_urls = Pretreatment.get_related_urls(head, text_number)
        # print(text_related_urls)
        # related_texts = [text]
        # for text_related_url in text_related_urls:
        #     related_texts.append(Pretreatment.split_txt(text_related_url)['text'])
        # print(related_texts)
        # similarity['text'] = SimilarityFromBERT.get_similarity(related_texts)[:1][0][1:]
        # print(similarity['text'])
        # 段落相似度
        paragraphs_similarity = []
        if EDU:
            for paragraph in paragraphs:
                for sentence in demo.get_EDUs(paragraph):
                    if sentence != "" and len(sentence) > sentence_lenth_limit:
                        print("开始搜索句子[{}]".format(sentence))
                        related_paragraphs = Pretreatment.get_related_paragraphs(sentence, paragraph_number)
                        paragraphs_similarity.extend(
                            SimilarityFromBERT.get_similarity([paragraph] + related_paragraphs)[:1][0][1:])
                        print(paragraphs_similarity)
            similarity['paragraph'] = paragraphs_similarity
        else:
            for paragraph in paragraphs:
                for sentence in re.split("[,.，。]", paragraph):
                    if sentence != "" and len(sentence) > sentence_lenth_limit:
                        print("开始搜索句子[{}]".format(sentence))
                        related_paragraphs = Pretreatment.get_related_paragraphs(sentence, paragraph_number)
                        paragraphs_similarity.extend(
                            SimilarityFromBERT.get_similarity([paragraph] + related_paragraphs)[:1][0][1:])
                        print(paragraphs_similarity)
            similarity['paragraph'] = paragraphs_similarity
        # 句子相似度
        # sentences_similarity = []
        # for sentence in sentences:
        #     if sentence != "" and len(sentence) > sentence_lenth_limit:
        #         print("开始搜索句子[{}]".format(sentence))
        #         related_sentences = Pretreatment.get_related_sentences(sentence, sentence_number)
        #         sentences_similarity.extend(SimilarityFromBERT.get_similarity([sentence]+related_sentences)[:1][0][1:])
        #         print(sentences_similarity)
        # similarity['sentence'] = sentences_similarity


if __name__ == '__main__':
    # # 对句子进行相似度计算
    # sentences = ["有一次使用到了contains和indexOf方法", "那什么时候使用Contains的上述方法", " contains方法的源码中其实是使用了IndexOf方法的,但效率还是有差别的",
    #              "contains和indexof都可以作为判断是否包含的方法"]
    # print(SimilarityFromBERT.get_similarity(sentences))

    # url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    # head_number = 20
    # text_number = 5
    # paragraph_number = 1
    # sentence_number = 1
    # SimilarityFromBERT.get_5d_similarities(url, head_number, text_number, paragraph_number, sentence_number, EDU=True)

    url = "https://blog.csdn.net/weixin_39664456/article/details/111175043"  # 待检测
    SimilarityFromBERT.get_similarity(["好", Pretreatment.split_txt(url)['sentences']])
