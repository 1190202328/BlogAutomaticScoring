import json
import random
import time
from pprint import pprint
import bs4
import jieba
from bs4 import BeautifulSoup
import requests
import re
from baiduspider import BaiduSpider

from src.Crawl import Crawl
from src.BERT import demo
from src.Pretreatment import Pretreatment
from src.SeparateCode import SeparateCode

url_pattern = dict()
pattern_csdn = "https://blog\\.csdn\\.net/.+/article/details/.+"
pattern_cnblogs = "https://www\\.cnblogs\\.com/.+/p/\\d+\\.html"
pattern_github = "https://.+\\.github\\.io/.+"
pattern_jianshu = "https://www\\.jianshu\\.com/p/.+"
pattern_csdn_main = "https://blog\\.csdn\\.net/.+"
pattern_cnblogs_main = "https://www\\.cnblogs\\.com/.+"
pattern_github_main = "https://.+\\.github\\.io"
pattern_jianshu_main = "https://www\\.jianshu\\.com/u/.+"
url_pattern['csdn'] = pattern_csdn
url_pattern['cnblogs'] = pattern_cnblogs
url_pattern['github'] = pattern_github
url_pattern['jianshu'] = pattern_jianshu
url_pattern['or'] = pattern_csdn + "|" + pattern_cnblogs + "|" + pattern_github + "|" + pattern_jianshu


def split_txt(txt_url, EDU=False, verbose=True):
    """
    根据url地址返回一个词典，词典中包含以下属性：1。head：标题；2。paragraphs：段落；3。sentences：句子；4。codes：代码；
    5。date：日期；6。text：全文（不含代码段）；
    :param EDU: 是否采用EDU来划分句子
    :param txt_url: url地址
    :return: 词典，如果不满足目的url（1。csdn2。cnblogs3。github4。简书），则返回None
    """
    result = dict()
    sentences = list()
    codes = list()
    clean_paragraphs = list()
    clean_text = ""
    head = ""
    text = ""
    update_date = ""
    clean_text_for_EDU = list()
    clean_text_for_EDU_element = ""

    is_illegal = False
    url = Pretreatment.get_real_url(txt_url, verbose=verbose)
    html = Pretreatment.get_raw_html(url, verbose=verbose)
    if html == "":
        return None
    bf = BeautifulSoup(html, "html.parser")

    if re.match(url_pattern['csdn'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="title-article")
        if content is None:
            if verbose:
                print("这个url标题有问题：" + txt_url)
            return None
        head = content.text.replace("\n", "")
        # text
        for child in bf.find("div", id="content_views"):
            if child.string is not None:
                text += child.string
            else:
                text += child.get_text(separator="\n")
        # date
        update_date = bf.find("span", class_="time").text[0:10]
    if re.match(url_pattern['cnblogs'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="postTitle")
        if content is None:
            content = bf.find("a", id="cb_post_title_url")
            if content is None:
                if verbose:
                    print("这个url标题有问题：" + txt_url)
                return None
        head = content.text.replace("\n", "")
        # text
        text = bf.find("div", id="cnblogs_post_body").getText()
        # for child in bf.find("div", id="cnblogs_post_body"):
        #     if child.name != "pre":
        #         if child.string is not None:
        #             text += child.string
        #         else:
        #             text += child.get_text(separator="\n")
        # date
        update_date = bf.find("span", id="post-date").text[0:10]
        # codes
        contents = bf.find_all("pre")
        for content in contents:
            codes.append(content.getText())
    if re.match(url_pattern['github'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="post-title")
        if content is None:
            content = bf.find("h1", class_="article-title sea-center")
            if content is None:
                content = bf.find("h1", class_="article-title")
                if content is None:
                    if verbose:
                        print("这个url标题有问题：" + txt_url)
                    return None
        head = content.text.replace("\n", "")
        # text
        text = bf.find("div", itemprop="articleBody").getText()
        # date
        update_date = bf.find("time").attrs['datetime'][0:10]
        # codes
        contents = bf.find_all("pre")
        digits = list()
        delete_codes = list()
        for content in contents:
            if re.match("\\d+", content.getText()):
                digits.append(content.getText())
                continue
            delete_codes.append(content.getText())
            raw_code = ""
            for child in content.children:
                if child.name == "span":
                    if child.string is not None:
                        raw_code += child.string + "\n"
            codes.append(raw_code)
        for digit in digits:
            start = text.find(digit)
            if start != -1:
                text = text[0:start] + text[start + len(digit):]
        for delete_code in delete_codes:
            start = text.find(delete_code)
            if start != -1:
                text = text[0:start] + text[start + len(delete_code):]
    if re.match(url_pattern['jianshu'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="_1RuRku")
        if content is None:
            if verbose:
                print("这个url标题有问题：" + txt_url)
            return None
        head = content.text.replace("\n", "")
        # text
        text = bf.find("article", class_="_2rhmJa").getText()
        # date
        update_date = ""
        # codes
        contents = bf.find_all("pre")
        for content in contents:
            codes.append(content.getText())

    if is_illegal:
        text = re.sub("(\\xa0)|(\\u200b)|(\\u2003)|(\\u3000)", "", text)
        text = re.sub("[\\t ]+", " ", text)
        text = re.sub("\n+", "\n", text)
        text = re.sub("(\n +)|( +\n)", "\n", text)
        to_search_code_text = Pretreatment.clean_code_for_text(text)
        more_codes = SeparateCode.get_codes(to_search_code_text)
        if more_codes:
            for more_code in more_codes:
                start = text.find(more_code)
                if start != -1:
                    text = text[0:start] + text[start + len(more_code):]
            codes += more_codes
        text = re.sub("\n+", "\n", text)
        # paragraphs
        paragraphs = text.split("\n")
        lenth = 200
        for paragraph in paragraphs:
            paragraph = re.sub("\\s+", " ", paragraph)
            if paragraph != " " and len(paragraph) > 2:
                clean_paragraphs.append(paragraph)
                if len(clean_text_for_EDU_element) >= lenth:
                    clean_text_for_EDU.append(clean_text_for_EDU_element)
                    clean_text_for_EDU_element = ""
                if paragraph[-1] in [",", ".", "。", "，", ":", "：", "、", "；", ";"]:
                    clean_text += paragraph
                    clean_text_for_EDU_element += paragraph
                else:
                    clean_text += paragraph + "。"
                    clean_text_for_EDU_element += paragraph + "。"
        text = ""
        for clean_paragraph in clean_paragraphs:
            text += clean_paragraph + "\n"
        # sentences
        if EDU:
            if clean_text_for_EDU_element != "":
                clean_text_for_EDU.append(clean_text_for_EDU_element)
            total_num = len(clean_text_for_EDU)
            j = 1
            for text in clean_text_for_EDU:
                if verbose:
                    print("第{}小篇(共{}小篇)".format(j, total_num))
                j += 1
                local_sentences = demo.get_EDUs(text)
                sentences.extend(local_sentences)
                pprint(local_sentences)
        else:
            raw_sentences = re.split("[。]", clean_text)
            for sentence in raw_sentences:
                if sentence != "":
                    if len(sentence) > 30:
                        sentences.extend(sentence.split("，"))
                    else:
                        sentences.append(sentence)
        result['head'] = head
        result['paragraphs'] = clean_paragraphs
        result['sentences'] = sentences
        result['codes'] = codes
        result['date'] = update_date
        result['text'] = text
        return result
    else:
        return None


if __name__ == '__main__':
    # 直接将文章完全喂给代码检测函数来检测
    url = "https://blog.csdn.net/Louis210/article/details/117415546"
    # url = "https://www.cnblogs.com/yuyueq/p/15119512.html"
    # url = "https://starlooo.github.io/2021/07/02/CaiKeng/"
    # url = "https://www.jianshu.com/p/92373a603d42"
    #
    # url = "https://blog.csdn.net/Louis210/article/details/119666026"
    # url = "https://blog.csdn.net/Baigker/article/details/118353220"
    # url = "https://blog.csdn.net/weixin_46219578/article/details/117462868"
    # url = "https://blog.csdn.net/m0_51250400/article/details/118405807"
    # url = "https://blog.csdn.net/buckbarnes/article/details/118547420"
    # url = "https://bit-ranger.github.io/blog/java/effective-java/"
    # url = "https://blog.csdn.net/z741481546/article/details/93514166"
    print("---------url>>>" + url)
    similarity = split_txt(url)
    print("---------head---------")
    print(similarity['head'])
    print("---------text---------")
    print(similarity['text'])
    # print(similarity['text'].encode('unicode_escape').decode())
    print("---------paragraphs共{}个---------".format(len(similarity['paragraphs'])))
    pprint(similarity['paragraphs'])
    print("---------sentences共{}个---------".format(len(similarity['sentences'])))
    pprint(similarity['sentences'])
    print("---------codes共{}个---------".format(len(similarity['codes'])))
    i = 1
    for code in similarity['codes']:
        print("-----------code{}-----------".format(i))
        i += 1
        print(code)
    print("---------date---------")
    print(similarity['date'])
