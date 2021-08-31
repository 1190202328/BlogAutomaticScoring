import json
import os
import re
from pprint import pprint
from colorama import Fore
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient
from tqdm import tqdm

from src.BERT import demo
from src.Pretreatment import Pretreatment
from src.SimilarityFromPMD import SimilarityFromPMD


def padding(result, padding_len, padding_content=0):
    if len(result) != 0:
        r = list(result)
        r = r[:padding_len] + [padding_content] * (padding_len - len(r))
        return r
    else:
        return [padding_content] * padding_len


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
        clean_sentences = Pretreatment.clean_with_low_frequency(sentences, my_stopwords)
        sentences = []
        for clean_sentence in clean_sentences:
            if len(clean_sentence) != 0:
                sentences.append(" ".join(clean_sentence))
        if len(sentences) <= 1:
            return [[0.0, 0.0], [0.0, 0.0]]
        # print("<<<<<<<<", end="")
        # print(sentences)
        return cosine_similarity(bc.encode(sentences))

    @staticmethod
    def get_text_related(contents, text, limit=0.80, verbose=True):
        """
        使用BERT，根据contents中的content与text的相似度来排除一些不相关的content，比如（下课之后，我查询了...）
        :param limit: 阈值，相似度小于limit的content将会被过滤
        :param contents: 内容列表。（可以是句子或者段落）
        :param text: 文章。
        :return: 过滤后的句子或段落列表
        """
        related_contents = list()
        if not contents:
            return related_contents

        stopwords_file = open("../src/text/stopwords.txt")
        stopwords_string = stopwords_file.read()
        stopwords_file.close()
        my_stopwords = stopwords_string.split("\n")

        bc = BertClient(check_length=False)
        total_contents = [text] + contents
        clean_contents = Pretreatment.clean_with_low_frequency(total_contents, my_stopwords)
        sentences = []
        origin_sentences = []
        for i in range(len(clean_contents)):
            if len(clean_contents[i]) != 0:
                origin_sentences.append(total_contents[i])
                sentences.append(" ".join(clean_contents[i]))
        if len(sentences) <= 1:
            return related_contents
        similarities = cosine_similarity(bc.encode(sentences))[:1][0][1:]
        # if verbose:
        #     for i in range(len(similarities)):
        #         print(similarities[i], end="")
        #         print(">>>>>>>>>", end="")
        #         print(origin_sentences[i + 1])
        count = 1
        for i in range(len(similarities)):
            if similarities[i] >= limit:
                related_contents.append(origin_sentences[i + 1])
            else:
                if verbose:
                    print("[{}]>>>{}>>>".format(count, similarities[i]) + origin_sentences[i + 1])
                count += 1
        return related_contents

    @staticmethod
    def get_5d_similarities(url, head_number=10, text_number=10, paragraph_number=10, sentence_number=10,
                            code_number=10,
                            sentence_lenth_limit=5, EDU=False, result=None, verbose=True, pre_verbose=False, save=None):
        """
        通过url链接获取5个维度的相似度：1。标题 2。全文 3。段落 4。句子 5。代码（都进行了停用词处理）（搜索结果中不包含自己）
        :param pre_verbose: 选择是否输出预处理阶段杂多的信息:True:复杂输出;False：简单输出
        :param save: 是否保存中间找到的数据；如果不为None：以该存档save.txt为文件名保存为json格式
        :param verbose: 选择是否输出杂多的信息:True:复杂输出;False：简单输出
        :param result:  是否自己提供result，而不用现成求
        :param EDU: 是否使用EDU来划分句子
        :param sentence_lenth_limit: 句子最短长度限制
        :param code_number: 相关代码行数
        :param sentence_number: 查找相关句子的文章篇数
        :param paragraph_number: 查找相关段落的文章篇数
        :param text_number: 相关文章数
        :param head_number: 相关标题数
        :param url: url链接
        :return: 字典，head:标题对应相似度。text:全文对应相似度。paragraph:段落对应相似度。sentence:句子对应相似度。code:代码对应相似度
        """
        sentence_len = 80
        code_len = 60

        total_count = 2
        similarity = dict()
        num = "ukn"
        if not result:
            result = Pretreatment.split_txt(url, verbose=verbose)
        if not result:
            return None
        head = result['head']
        text = result['text']
        paragraphs = result['paragraphs']
        codes = result['codes']
        if save is not None:
            num = save
            search_result = dict()
            search_result['head'] = []
            search_result['text'] = []
            search_result['paragraphs'] = dict()
            search_result['sentences'] = dict()
            search_result['codes'] = dict()
        if verbose:
            print("不相关段落如下(小于0.80的)")
        paragraphs = SimilarityFromBERT.get_text_related(paragraphs, text, limit=0.80, verbose=verbose)
        to_search_sentences = []
        if pre_verbose:
            notice = tqdm(total=len(paragraphs), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
            notice.set_description("[{}] ".format(num) + url + " >>>Pre")
        i = 0
        for paragraph in paragraphs:
            if total_count - 2 >= sentence_len:
                break
            if EDU:
                sentences = demo.get_EDUs(paragraph)
            else:
                sentences = re.split("[,.，。]", paragraph)
            if verbose:
                print("不相关句子如下(小于0.85的)")
            if save is not None:
                search_result['paragraphs'][paragraph] = []
            sentences = SimilarityFromBERT.get_text_related(sentences, text, limit=0.85, verbose=verbose)
            clean_sentences = []
            for sentence in sentences:
                if sentence != "" and len(sentence) > sentence_lenth_limit and total_count - 2 < sentence_len:
                    if save is not None:
                        search_result['sentences'][sentence] = []
                    clean_sentences.append(sentence)
                    total_count += 1
            to_search_sentences.append(clean_sentences)
            if pre_verbose:
                notice.update(1)
            i += 1
        if pre_verbose:
            notice.close()
        to_search_codes = []
        if codes:
            for code in codes:
                if len(to_search_codes) >= code_len:
                    break
                lines = Pretreatment.clean_code(code)
                for line in lines:
                    if len(to_search_codes) >= code_len:
                        break
                    to_search_codes.append(line)
                    total_count += 1
        if not verbose:
            notice = tqdm(total=total_count, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET))
            notice.set_description("[{}] ".format(num) + url + " >>>")
        # 标题相似度
        related_heads, _ = Pretreatment.get_related_head(head, head_number, url=url, verbose=verbose)
        if save is not None:
            search_result['head'] = related_heads
        head_similarity = SimilarityFromBERT.get_similarity([head] + related_heads)[:1][0][1:]
        similarity['head'] = padding(head_similarity, head_number)
        if verbose:
            print(related_heads)
            print(len(similarity['head']))
            print(similarity['head'])
        else:
            notice.update(1)
        # 全文相似度
        related_texts, _ = Pretreatment.get_related_texts(head, text_number, url=url, verbose=verbose)
        if save is not None:
            search_result['text'] = related_texts
        related_texts = [text] + related_texts
        text_similarity = SimilarityFromBERT.get_similarity(related_texts)[:1][0][1:]
        similarity['text'] = padding(text_similarity, text_number)
        if verbose:
            print(similarity['text'])
        else:
            notice.update(1)
        # 段落和句子相似度
        paragraphs_similarity = []
        sentences_similarity = []
        invalid_count = 0
        for i in range(len(paragraphs)):
            if verbose:
                print("段落>>>" + paragraphs[i])
            for sentence in to_search_sentences[i]:
                if verbose:
                    print("开始搜索句子[{}]".format(sentence))
                related_paragraphs, related_sentences, _, invalid = Pretreatment.get_related_paragraphs_and_sentences \
                    (sentence, paragraph_number=paragraph_number, sentence_number=sentence_number, url=url,
                     verbose=verbose)
                if save is not None:
                    search_result['paragraphs'][paragraphs[i]] = related_paragraphs
                    search_result['sentences'][sentence] = related_sentences
                if invalid >= 2:
                    invalid_count += 1
                else:
                    invalid_count = 0
                if invalid_count >= 3:
                    notice.close()
                    return None
                paragraph_similarity = SimilarityFromBERT.get_similarity([paragraphs[i]] + related_paragraphs)[:1][0][
                                       1:]
                paragraph_similarity = padding(paragraph_similarity, paragraph_number)
                paragraphs_similarity.append(paragraph_similarity)
                if verbose:
                    print("------段落相似度------")
                    print(paragraph_similarity)
                sentence_similarity = SimilarityFromBERT.get_similarity([sentence] + related_sentences)[:1][0][1:]
                sentence_similarity = padding(sentence_similarity, sentence_number)
                sentences_similarity.append(sentence_similarity)
                if verbose:
                    print("------句子相似度------")
                    print(sentence_similarity)
                else:
                    notice.update(1)
        similarity['paragraph'] = paragraphs_similarity
        similarity['sentence'] = sentences_similarity
        # 代码相似度
        codes_similarity = []
        for line in to_search_codes:
            related_codes = Pretreatment.get_related_codes(line, code_number, verbose=verbose)
            if save is not None:
                search_result['codes'][line] = related_codes
            if verbose:
                print("相关codes如下:")
            code_similarity = []
            j = 1
            for related_code in related_codes:
                if verbose:
                    print("[{}]>>>".format(j) + related_code.__str__())
                j += 1
                if SimilarityFromPMD.is_similar(line, related_code):
                    if SimilarityFromBERT.get_similarity([line, related_code])[:1][0][1] < 0.05:
                        code_similarity.append(0)
                    else:
                        code_similarity.append(1)
                else:
                    if SimilarityFromBERT.get_similarity([line, related_code])[:1][0][1] > 0.95:
                        code_similarity.append(1)
                    else:
                        code_similarity.append(0)
            code_similarity = padding(code_similarity, code_number)
            codes_similarity.append(code_similarity)
            if verbose:
                print("------code相似度------")
                print(code_similarity)
            else:
                notice.update(1)
        similarity['code'] = codes_similarity
        if not verbose:
            notice.close()
        if save is not None:
            if os.path.exists('../src/text/存档{}.txt'.format(save)):
                print("\n" + '../src/text/存档{}.txt'.format(save) + ">>>已存在\n", end="")
            else:
                f = open('../src/text/存档{}.txt'.format(save), 'w')
                f.write(json.dumps(search_result, ensure_ascii=False, indent=2))
                f.close()
        return similarity

    @staticmethod
    def get_score(similarity, head_number, text_number, paragraph_number, sentence_number, code_number,
                  similarity_limit=0.9, is_contribute=0.9):
        """
        根据5个维度的相似度来评估分数。1。标题 2。全文 3。段落 4。句子 5。代码
        :param is_contribute: 原创比例大于similarity_limit的相才可以记为原创
        :param similarity_limit: 小于similarity_limit的相似度才可以算作原创
        :param similarity: 含有五个维度相似度的词典
        head:标题对应相似度。text:全文对应相似度。paragraph:段落对应相似度。sentence:句子对应相似度。code:代码对应相似度
        :return: 分数0，1，2，3，4，5
        """
        paragraph_ = 0
        sentence_ = 0
        code_ = 0
        i = 1
        for paragraph in similarity['paragraph']:
            count = 0
            for paragraph_similarity in paragraph:
                if paragraph_similarity < similarity_limit:
                    count += 1
            print("第{}个段落原创性占比>>>>>{}".format(i, count / paragraph_number))
            if (count / paragraph_number) > is_contribute:
                paragraph_ += 1
            i += 1
        i = 1
        for sentence in similarity['sentence']:
            count = 0
            for sentence_similarity in sentence:
                if sentence_similarity < similarity_limit:
                    count += 1
            print("第{}个句子原创性占比>>>>>{}".format(i, count / sentence_number))
            if (count / paragraph_number) > is_contribute:
                sentence_ += 1
            i += 1
        i = 1
        for code in similarity['code']:
            count = 0
            for code_similarity in code:
                if code_similarity < similarity_limit:
                    count += 1
            print("第{}行code原创性占比>>>>>{}".format(i, count / code_number))
            if (count / paragraph_number) > is_contribute:
                code_ += 1
            i += 1
        count = 0
        for head_similarity in similarity['head']:
            if head_similarity < similarity_limit:
                count += 1
        print("标题原创性占比>>>>>{}".format(count / head_number))
        count = 0
        for text_similarity in similarity['text']:
            if text_similarity < similarity_limit:
                count += 1
        print("全文原创性占比>>>>>{}".format(count / text_number))
        print("段落原创性占比>>>>>{}".format(paragraph_ / len(similarity['paragraph'])))
        print("句子原创性占比>>>>>{}".format(sentence_ / len(similarity['sentence'])))
        if similarity['code']:
            print("code原创性占比>>>>>{}".format(code_ / len(similarity['code'])))

    # 测试时使用，测试结束删掉


if __name__ == '__main__':
    # # 对句子进行相似度计算
    # sentences = ["有一次使用到了contains和indexOf方法", "那什么时候使用Contains的上述方法", " contains方法的源码中其实是使用了IndexOf方法的,但效率还是有差别的",
    #              "contains和indexof都可以作为判断是否包含的方法", "并且"]
    # print(SimilarityFromBERT.get_similarity(sentences))

    # url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    # url = "https://blog.csdn.net/Louis210/article/details/119666026"
    # url = "https://www.cnblogs.com/yuyueq/p/15119512.html"
    # url = "https://starlooo.github.io/2021/07/02/CaiKeng/"

    # url = "https://blog.csdn.net/zhuzyibooooo/article/details/118527726?spm=1001.2014.3001.5501"
    # url = "https://blog.csdn.net/Louis210/article/details/119666026?spm=1001.2014.3001.5501"
    # url = "https://blog.csdn.net/weixin_46219578/article/details/117462868"
    url = "https://blog.csdn.net/Louis210/article/details/119649950"
    head_number = 3
    text_number = 3
    paragraph_number = 3
    sentence_number = 3
    code_number = 3
    result = SimilarityFromBERT.get_5d_similarities(url, head_number, text_number, paragraph_number,
                                                    sentence_number,
                                                    code_number,
                                                    EDU=False,
                                                    verbose=True,
                                                    pre_verbose=False,
                                                    save=0
                                                    )
    print("-----------标题相似度---------")
    pprint(result['head'])
    print("-----------全文相似度---------")
    pprint(result['text'])
    print("-----------段落相似度---------")
    pprint(result['paragraph'])
    print("-----------句子相似度---------")
    pprint(result['sentence'])
    print("-----------code相似度---------")
    pprint(result['code'])

    with open('../src/text/存档0.txt', 'r') as f:
        result = json.loads(f.read())
        print("<<<head>>>")
        print(result['head'])
        print("<<<text>>>")
        print(result['text'])
        print("<<<paragraphs>>>")
        for paragraph in result['paragraphs']:
            print(paragraph)
            print(result['paragraphs'][paragraph])
        print("<<<sentences>>>")
        for sentence in result['sentences']:
            print(sentence)
            print(result['sentences'][sentence])
        print("<<<codes>>>")
        for code in result['codes']:
            print(code)
            print(result['codes'][code])
