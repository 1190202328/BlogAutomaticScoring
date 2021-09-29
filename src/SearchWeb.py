import json
import random
import re
from pprint import pprint
from typing import Union
import bs4
from bs4 import BeautifulSoup
from datetime import datetime
import time

from src import HTML, GetWebResource, Clean
from src import Global


def clean_to_search_keywords(keywords: str) -> str:
    """
    清理需要搜索的句子
    :param keywords: 原句子
    :return: 新的干净句子
    """
    keywords = re.sub('(#+)|(%+)', '', keywords).strip().lower()
    keywords = re.sub('\\s+', ' ', keywords)
    return keywords


def get_next_baidu_url(baidu_html: str) -> str:
    """
    通过源百度的html获取下一页搜索页面的百度url
    :param baidu_html: 源百度html
    :return: 下一页搜索页面的百度url
    """
    bf = BeautifulSoup(baidu_html, "html.parser")
    url = bf.find("a", class_="n")
    if url is not None:
        return "https://www.baidu.com" + url.attrs['href']
    return ""


def get_related_head_and_text(text_head: str, update_time: str, head_number: int = 10,
                              text_number: int = 10, page_limit: int = 3, url: str = "", verbose: bool = True) \
        -> Union[tuple[list, list], tuple[None, None]]:
    """
    根据标题在百度搜索time之前的相关文章，取出前head_number篇文章的标题以及text_number篇文章的全文
    :param text_number: 需要相关全文的个数
    :param head_number: 需要相关标题的个数
    :param verbose: 选择是否输出杂多的信息:True:复杂输出;False：简单输出
    :param update_time: 发布时间
    :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
    :param page_limit: 最大页码数
    :param text_head: 文章标题
    :return: head_number篇文章的标题,text_number篇文章的全文
    """
    text_head = clean_to_search_keywords(text_head)
    if text_head == '':
        return None, None
    pn = 0
    update_time = datetime.strptime(update_time, "%Y-%m-%d")
    total_urls = list()
    invalid = 0
    related_heads = list()
    related_texts = list()
    baidu_url = 'http://baidu.com/s?wd=' + text_head + "&rn=50" + "&oq=" + text_head + "&ie=utf-8"
    i = 1
    while True:
        pn += 1
        if pn > page_limit:
            break
        if len(related_heads) >= head_number and len(related_texts) >= text_number:
            return related_heads, related_texts
        article_urls = list()
        html = HTML.get_raw_html(baidu_url, verbose=verbose)
        pre_baidu_url = baidu_url
        baidu_url = get_next_baidu_url(html)
        if verbose:
            print("第{}页".format(pn))
        if baidu_url == "":
            baidu_url = 'http://baidu.com/s?wd=' + text_head + "&pn=" + str(
                pn * 50) + "&rn=50" + "&oq=" + text_head + "&ie=utf-8"
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("div", class_="c-container")
        for content in contents:
            if len(related_heads) >= head_number and len(related_texts) >= text_number:
                return related_heads, related_texts
            real_url = ""
            flag = False
            for child in content.children:
                if isinstance(child, bs4.element.NavigableString):
                    continue
                for c in child.children:
                    if c.name == "a" and c.parent.name == "h3":
                        real_url = HTML.get_real_url(c.attrs['href'], verbose=verbose)
                        if real_url != "":
                            flag = True
                            article_urls.append(real_url)
                            break
            if flag:
                if real_url in total_urls:
                    continue
                if url.find(real_url) != -1:
                    continue
                if not re.match(Global.url_pattern['or'], real_url):
                    continue
                total_urls.append(real_url)
                result = GetWebResource.split_txt(real_url, verbose=verbose)
                if result is None:
                    continue
                print(real_url)
                update_time_this = datetime.strptime(result['date'], "%Y-%m-%d")
                if update_time_this > update_time:
                    continue
                related_head = result['head']
                related_text = result['text']
                if len(related_heads) < head_number:
                    related_heads.append(related_head)
                    if verbose:
                        print('第{}个标题>>>'.format(i), related_head)
                if len(related_texts) < text_number:
                    related_texts.append(related_text)
                    if verbose:
                        print('第{}篇全文：'.format(i))
                        print(related_text)
                i += 1
        if verbose:
            print("url共有{}个".format(len(article_urls)))
        if len(article_urls) == 0:
            print("\n此url第{}次暂时无法访问>>>{}\n".format(invalid + 1, pre_baidu_url), end="")
            baidu_url = pre_baidu_url
            if invalid == 1:
                pn -= 1
                time.sleep(random.randrange(100, 200, 1))
            elif invalid == 2:
                time.sleep(random.randrange(200, 300, 1))
            else:
                time.sleep(random.randrange(100, 200, 1))
                return None, None
            invalid += 1
    return related_heads, related_texts


def get_related_codes(code: str, number: int, limit: int = 7, verbose: bool = True) -> []:
    """
    根据code获取相关的code
    :param verbose: 是否繁杂输出
    :param limit: 每行代码的最短长度，小于该长度的代码行将会被过滤
    :param code: 一行源代码
    :param number: 需要获取相关code的数量，最多100行相关的code
    :return: 相关code的列表(code中不含中文注释)
    """
    if verbose:
        print("开始搜索：" + code)
    count = 0
    related_codes = []
    api = "https://searchcode.com/api/codesearch_I/?q=" + code + "&p=0&per_page=100"
    text = HTML.get_raw_html_origin(api)
    if text == "":
        return related_codes
    api_result = json.loads(text)
    rs = api_result.get('results')
    if rs is None:
        return related_codes
    for result in rs:
        lines = result['lines']
        # print(lines)
        ids = list()
        for id in lines:
            ids.append(id)
        for i in range(1, len(ids) - 1):
            if int(ids[i]) - 1 == int(ids[i - 1]) and int(ids[i]) + 1 == int(ids[i + 1]):
                clean_lines = Clean.clean_code(lines[ids[i]], limit)
                for clean_line in clean_lines:
                    try:
                        clean_line = eval(clean_line + "' '")
                        clean_line = clean_line.replace("\n", "")
                    except Exception as e:
                        if verbose:
                            print(e.args)
                        pass
                    related_codes.append(clean_line.__str__())
                    count += 1
                    if count >= number:
                        return related_codes
    return related_codes


def get_related_paragraphs_and_sentences(original_sentence: str, paragraph_number: int = 5, sentence_number: int = 10,
                                         page_limit: int = 2,
                                         url: str = "",
                                         verbose: bool = True) -> ([], [], bool, int):
    """
    在百度上获取相关的句子
    :param verbose: 是否繁杂输出
    :param sentence_number: 需要搜索的相关句子的数
    :param paragraph_number: 需要搜索的相关段落的数
    :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
    :param page_limit: 百度搜索的页码限制
    :param original_sentence: 源句子
    :return: 相关段落的列表，相关句子的列表，find, invalid（如果为2则表示此次查找ip被封掉，失败）
    """
    original_sentence = clean_to_search_keywords(original_sentence)
    pn = 0
    article_count = 1
    total_urls = list()
    find = False
    invalid = 0
    related_sentences = list()
    related_paragraphs = list()
    baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&rn=50" + "&oq=" + original_sentence + "&ie=utf-8"
    while True:
        pn += 1
        if pn > page_limit:
            break
        if len(related_paragraphs) >= paragraph_number and len(related_sentences) >= sentence_number:
            return related_paragraphs, related_sentences, find, invalid
        article_urls = list()
        html = HTML.get_raw_html(baidu_url, verbose=verbose)
        bf = BeautifulSoup(html, "html.parser")
        if re.match(Global.not_find, bf.get_text(), flags=re.S):
            return [], [], False, 0
        pre_baidu_url = baidu_url
        baidu_url = get_next_baidu_url(html)
        if verbose:
            print("第{}页".format(pn))
        if baidu_url == "":
            baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&pn=" + str(
                pn * 50) + "&rn=50" + "&oq=" + original_sentence + "&ie=utf-8"
        contents = bf.find_all("div", class_="c-container")
        for content in contents:
            if len(related_paragraphs) >= paragraph_number and len(related_sentences) >= sentence_number:
                return related_paragraphs, related_sentences, find, invalid
            real_url = ""
            red_strings = set()
            flag = False
            for child in content.children:
                if isinstance(child, bs4.element.NavigableString):
                    continue
                for c in child.children:
                    if c.name == "a" and c.parent.name == "h3":
                        real_url = HTML.get_real_url(c.attrs['href'], verbose=verbose)
                        if real_url != "":
                            flag = True
                            article_urls.append(real_url)
                            break
            if flag:
                if real_url in total_urls:
                    continue
                if url.find(real_url) != -1:
                    find = True
                    continue
                if not re.match(Global.url_pattern['or'], real_url):
                    continue
                for child in content.descendants:
                    if child.name == "em" and child.string != "":
                        red_strings.add(child.string)
                total_urls.append(real_url)
                if verbose:
                    print("第{}篇文章>>>>>>>>>".format(article_count) + real_url + ">>>>", end="")
                    print(red_strings)
                article_count += 1
                article_paragraphs = list()
                result = GetWebResource.split_txt(real_url, verbose=verbose)
                if result is None:
                    continue
                paragraphs = result.get('paragraphs')
                if paragraphs is not None:
                    for paragraph in paragraphs:
                        for substring in red_strings:
                            if paragraph != "" and paragraph.find(substring) != -1 and len(
                                    related_paragraphs) < paragraph_number:
                                related_paragraphs.append(paragraph)
                                article_paragraphs.append(paragraph)
                                break
                if verbose:
                    if article_paragraphs:
                        print("段落如下：")
                        j = 1
                        for article_paragraph in article_paragraphs:
                            print("[{}]>>>".format(j) + article_paragraph)
                            j += 1
                article_sentences = list()
                sentences = result.get('sentences')
                if sentences is not None:
                    for sentence in sentences:
                        for substring in red_strings:
                            if sentence != "" and sentence.find(substring) != -1 and len(
                                    related_sentences) < sentence_number:
                                related_sentences.append(sentence)
                                article_sentences.append(sentence)
                                break
                if verbose:
                    if article_sentences:
                        print("句子如下：")
                        j = 1
                        for article_sentence in article_sentences:
                            print("[{}]>>>".format(j) + article_sentence)
                            j += 1
        if verbose:
            print("url共有{}个".format(len(article_urls)))
        if len(article_urls) == 0:
            print("\n此url第{}次暂时无法访问>>>{}\n".format(invalid + 1, pre_baidu_url), end="")
            baidu_url = pre_baidu_url
            if invalid == 1:
                pn -= 1
                time.sleep(random.randrange(100, 200, 1))
            elif invalid == 2:
                time.sleep(random.randrange(200, 300, 1))
            else:
                time.sleep(random.randrange(100, 200, 1))
            invalid += 1
    return related_paragraphs, related_sentences, find, invalid


if __name__ == '__main__':
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
    # url = "https://blog.csdn.net/cobracanary/article/details/119612536"


    print("---------url>>>" + url)
    result = GetWebResource.split_txt(url)
    head = result['head']
    text = result['text']
    paragraphs = result['paragraphs']
    sentences = result['sentences']
    codes = result['codes']
    update_date = result['date']
    print("---------head---------")
    print(head)
    print("---------text---------")
    print(text)
    # print(similarity['text'].encode('unicode_escape').decode())
    print("---------paragraphs共{}个---------".format(len(paragraphs)))
    pprint(paragraphs)
    print("---------sentences共{}个---------".format(len(sentences)))
    pprint(sentences)
    print("---------codes共{}个---------".format(len(codes)))
    i = 1
    for code in codes:
        print("-----------code{}-----------".format(i))
        i += 1
        print(code)
    print("---------date---------")
    print(update_date)
    print("")

    get_related_head_and_text(head, update_date, url=url)
