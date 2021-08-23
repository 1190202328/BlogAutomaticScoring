import json
from pprint import pprint

import bs4
import jieba
from bs4 import BeautifulSoup
import requests
import re
from datetime import date
from baiduspider import BaiduSpider
from tqdm import tqdm

from src.BERT import demo
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


class Pretreatment:
    """
    预处理类，负责从网页抓取文档，预处理文档。
    """

    @staticmethod
    def split_txt(txt_url, EDU=False):
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
        url = Pretreatment.get_real_url(txt_url)
        html = Pretreatment.get_raw_html(url)
        if html == "":
            return None
        bf = BeautifulSoup(html, "html.parser")

        if re.match(url_pattern['csdn'], url):
            is_illegal = True
            # head
            content = bf.find("h1", class_="title-article")
            if content is None:
                print("这个url标题有问题：" + txt_url)
                return None
            head = content.text.replace("\n", "")
            # text
            # text = bf.find("div", id="content_views").get_text()
            for child in bf.find("div", id="content_views"):
                if child.name != "pre":
                    if child.string is not None:
                        text += child.string
                    else:
                        text += child.get_text(separator="\n")
            # date
            update_date = bf.find("span", class_="time").text[0:10]
            # codes
            contents = bf.find_all("pre")
            for content in contents:
                codes.append(content.getText())
        if re.match(url_pattern['cnblogs'], url):
            is_illegal = True
            # head
            content = bf.find("h1", class_="postTitle")
            if content is None:
                content = bf.find("a", id="cb_post_title_url")
                if content is None:
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
            for code in codes:
                start = text.find(code)
                if start != -1:
                    text = text[0:start] + text[start + len(code):]
            text = re.sub("(\\xa0)|(\\u200b)|(\\u2003)|(\\u3000)", "", text)
            text = re.sub("[\\t ]+", " ", text)
            text = re.sub("\n+", "\n", text)
            text = re.sub("(\n +)|( +\n)", "\n", text)
            more_codes = SeparateCode.get_codes(text)
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

    @staticmethod
    def get_related_head(text_head, number, page_limit=10, url="", verbose=True):
        """
        根据标题在百度搜索相关文章，取出前number篇文章的标题
        :param verbose: 选择是否输出杂多的信息:True:复杂输出;False：简单输出
        :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
        :param page_limit: 最大页码数
        :param number: 需要相关文章的篇数
        :param text_head: 文章标题
        :return: number篇文章的标题,find
        """
        total_urls = list()
        total_titles = list()
        count = 0
        pn = 1
        find = False
        while True:
            results = BaiduSpider().search_web(text_head, pn=pn, exclude=['all']).get('results')
            if verbose:
                print("pn = {}".format(pn))
            pn += 1
            if pn > page_limit:
                return total_titles, find
            for result in results:
                if verbose:
                    print(result)
                if count >= number:
                    break
                if result.get('title') is None:
                    continue
                title = result.get('title').split("_")
                if len(title) == 1:
                    title = result.get('title').split("-")
                real_url = Pretreatment.get_real_url(result.get('url'))
                if real_url in total_urls:
                    continue
                if url.find(real_url) != -1:
                    find = True
                    continue
                total_titles.append(title[0])
                total_urls.append(real_url)
                if verbose:
                    print("count = {}".format(count))
                count += 1
            if count >= number:
                break
        return total_titles, find

    @staticmethod
    def get_urls(main_url):
        """
        根据学生主页面获取所有博客的url地址
        :param main_url: 主页面地址，包括（1。csdn2。cnblogs3。github4。简书）
        :return: 所有博客的ulr地址
        """
        html = Pretreatment.get_raw_html(main_url)
        bf = BeautifulSoup(html, "html.parser")
        urls = set()
        contents = bf.find_all("a")
        if re.match(pattern_csdn_main, main_url):
            for content in contents:
                if content.get("href") is not None and re.match(".*/article/details.*", content.get("href")):
                    if re.match(".*#comments|.*blogdevteam.*", content.get("href")):
                        continue
                    urls.add(content.get("href"))
        if re.match(pattern_cnblogs_main, main_url):
            for content in contents:
                if content.get("href") is not None and re.match(pattern_cnblogs, content.get("href")):
                    urls.add(content.get("href"))
        if re.match(pattern_github_main, main_url):
            for content in contents:
                if content.get("href") is not None and re.match("/\\d{4}/\\d{2}/\\d{2}/.+/", content.get("href")):
                    urls.add(main_url + content.get("href"))
        if re.match(pattern_jianshu_main, main_url):
            for content in contents:
                if content.get("href") is not None and re.match("/p/\\w+", content.get("href")):
                    if not re.match(".*#comments.*", content.get("href")):
                        urls.add("https://www.jianshu.com" + content.get("href"))
        return urls

    @staticmethod
    def get_main_url(url):
        """
        根据url地址返回主页的url地址
        :param url: url地址（1。csdn2。cnblogs3。github4。简书）
        :return: 主页的URL地址，如果找不到则返回""
        """
        main_url = ""
        if re.match(pattern_csdn_main, url):
            temps = url.split("/")
            main_url = "https://blog.csdn.net/" + temps[3]
        if re.match(pattern_cnblogs_main, url):
            temps = url.split("/")
            main_url = "https://www.cnblogs.com/" + temps[3]
        if re.match(pattern_github_main, url):
            temps = url.split("/")
            main_url = "https://" + temps[2]
        if re.match(pattern_jianshu, url):
            html = Pretreatment.get_raw_html(url)
            bf = BeautifulSoup(html, "html.parser")
            contents = bf.find_all("a", href=True)
            for content in contents:
                if re.match("/u/.+", content.get("href")):
                    main_url = "https://www.jianshu.com" + content.get("href")
                    break
        return main_url

    @staticmethod
    def get_related_texts(text_head, number, page_limit=10, url="", verbose=True):
        """
        根据text_head在百度搜索，取出前number篇文章
        :param page_limit: 百度搜索的最大页码限制
        :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
        :param number: 需要相关文章的篇数
        :param text_head: 搜索的标题
        :return: number篇文章,find
        """
        total_urls = list()
        related_texts = list()
        find = False
        count = 0
        pn = 1
        while True:
            results = BaiduSpider().search_web(text_head, pn=pn, exclude=['all']).get('results')
            if verbose:
                print("pn = {}".format(pn))
            pn += 1
            if pn > page_limit:
                break
            for result in results:
                if count >= number:
                    break
                if result.get('url') is None:
                    continue
                real_url = Pretreatment.get_real_url(result.get('url'))
                if real_url in total_urls:
                    continue
                if url.find(real_url) != -1:
                    find = True
                    continue
                if re.match(url_pattern['or'], real_url):
                    result = Pretreatment.split_txt(real_url)
                    if result is not None:
                        related_texts.append(result['text'])
                        total_urls.append(real_url)
                        count += 1
                        if verbose:
                            print("count = {}".format(count))
            if count >= number:
                break
        if verbose:
            print(total_urls)
        return related_texts, find

    @staticmethod
    def get_raw_html(url):
        """
        根据url获取html文档
        :param url: url地址
        :return: html文档
        """
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, compress',
            'Accept-Language': 'en-us;q=0.5,en;q=0.3',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
        }
        try:
            r = requests.get(url=url, headers=headers, timeout=10)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(e.args)
            return ""

    @staticmethod
    def get_real_url(url):
        """
        根据url获得真实的url地址
        :param url: 源url地址
        :return: 真实url地址
        """
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, compress',
            'Accept-Language': 'en-us;q=0.5,en;q=0.3',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
        }
        try:
            r = requests.get(url=url, headers=headers, timeout=10)
            r.raise_for_status()
            return r.url
        except Exception as e:
            print(e.args)
            return ""

    @staticmethod
    def get_next_baidu_url(baidu_html):
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

    @staticmethod
    def get_related_codes(code, number, limit=7, verbose=True):
        """
        根据code获取相关的code
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
        text = Pretreatment.get_raw_html(api)
        if text == "":
            return related_codes
        api_result = json.loads(text)
        results = api_result.get('results')
        if results is None:
            return related_codes
        for result in results:
            lines = result['lines']
            # print(lines)
            ids = list()
            for id in lines:
                ids.append(id)
            for i in range(1, len(ids) - 1):
                if int(ids[i]) - 1 == int(ids[i - 1]) and int(ids[i]) + 1 == int(ids[i + 1]):
                    clean_lines = Pretreatment.clean_code(lines[ids[i]], limit)
                    for clean_line in clean_lines:
                        try:
                            clean_line = eval(clean_line + "' '")
                            clean_line = clean_line.replace("\n", "")
                        except:
                            pass
                        related_codes.append(clean_line)
                        count += 1
                        if count >= number:
                            return related_codes
        return related_codes

    @staticmethod
    def clean_code(code, limit=7):
        """
        获得干净的codes列表（每一个元素为一行代码）（不含\t，\n，连续2个以上空格，注释，不含import,include等）
        目前注释只支持
        单行注释：//和#
        多行注释：/* */和三个'和三个"
        :param limit: 每行代码的最短长度，小于该长度的代码行将会被过滤
        :param code: 源代码段
        :return: codes列表
        """
        blank_pattern = "[\\t\\n]+"
        codes = list()
        code = re.sub("/(\\*).*?(\\*)/", "", code, flags=re.S)
        code = re.sub("'''.*?'''", "", code, flags=re.S)
        code = re.sub('""".*?"""', "", code, flags=re.S)
        lines = code.split("\n")
        # print("-------清除之后-------")
        # pprint(lines)
        for line in lines:
            line = re.sub(blank_pattern, "", line)
            line = re.sub(" +", " ", line)
            java_start = line.find("//")
            python_start = line.find("#")
            if java_start != -1:
                line = line[0:java_start]
            if python_start != -1:
                line = line[0:python_start]
            # line = re.sub(r'[\u4e00-\u9fa5].*[\u4e00-\u9fa5]', "", line, flags=re.S)
            if len(line) < limit or re.match("(import .*)|(include .*)|(from .*)", line):
                continue
            codes.append(line)
        return codes

    @staticmethod
    def get_related_paragraphs_and_sentences(original_sentence, paragraph_number=5, sentence_number=10,
                                             page_limit=2,
                                             url="",
                                             verbose=True):
        """
        在百度上获取相关的句子
        :param sentence_number: 需要搜索的相关句子的数
        :param paragraph_number: 需要搜索的相关段落的数
        :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
        :param page_limit: 百度搜索的页码限制
        :param original_sentence: 源句子
        :return: 相关段落的列表，相关句子的列表，find, invalid（如果为2则表示此次查找ip被封掉，失败）
        """
        pn = 0
        article_count = 1
        total_urls = list()
        find = False
        invalid = 0
        related_sentences = list()
        related_paragraphs = list()
        baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&rn=50" + "&oq=" + original_sentence + "&ie=utf-8"
        while True:
            if len(related_paragraphs) >= paragraph_number and len(related_sentences) >= sentence_number:
                return related_paragraphs, related_sentences, find, invalid
            article_urls = list()
            html = Pretreatment.get_raw_html(baidu_url)
            baidu_url = Pretreatment.get_next_baidu_url(html)
            if verbose:
                print("第{}页".format(pn + 1))
            pn += 1
            if baidu_url == "":
                baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&pn=" + str(
                    pn * 50) + "&rn=50" + "&oq=" + original_sentence + "&ie=utf-8"
            if pn > page_limit:
                break
            bf = BeautifulSoup(html, "html.parser")
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
                            real_url = Pretreatment.get_real_url(c.attrs['href'])
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
                    if not re.match(url_pattern['or'], real_url):
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
                    result = Pretreatment.split_txt(real_url)
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
                                print("[{}]>>>".format(j)+article_paragraph)
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
                invalid += 1
        return related_paragraphs, related_sentences, find, invalid

    @staticmethod
    def clean_with_low_frequency(documents, stopwords_set=""):
        """
        将按列表存储的文档进行清洗
        :param stopwords_set: 可选参数, 如果选上，则表示自己提供停用词
        :param documents: 按列表存储的文档，列表中一个元素为一个文档
        :return: 清洗好的文档，二维列表，一行为一个文档的清洗后的词
        """
        if stopwords_set:
            my_stopwords = stopwords_set
        else:
            stopwords_file = open("./src/text/stopwords.txt")
            stopwords_string = stopwords_file.read()
            stopwords_file.close()
            my_stopwords = stopwords_string.split("\n")
        texts = list()
        for document in documents:
            text = list()
            for word in jieba.cut(document):
                word = word.lower().strip()
                if (word in my_stopwords) or re.match("\\s+", word) or re.match("\\d+", word):
                    continue
                text.append(word)
            texts.append(text)
        return texts


if __name__ == '__main__':
    # url = "https://blog.csdn.net/Louis210/article/details/117415546"
    # url = "https://www.cnblogs.com/yuyueq/p/15119512.html"
    # url = "https://starlooo.github.io/2021/07/02/CaiKeng/"
    # url = "https://www.jianshu.com/p/92373a603d42"
    #
    # url = "https://blog.csdn.net/Louis210/article/details/119666026"
    url = "https://blog.csdn.net/Baigker/article/details/118353220"
    # url = "https://blog.csdn.net/weixin_46219578/article/details/117462868"
    # url = "https://blog.csdn.net/m0_51250400/article/details/118405807"
    # url = "https://blog.csdn.net/buckbarnes/article/details/118547420"
    print("---------url>>>" + url)
    similarity = Pretreatment.split_txt(url)
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

    # similarity = Pretreatment.split_txt(url, EDU=True)
    # print("---------EDU-sentences--------")
    # pprint(similarity['sentences'])

    # url = "https://blog.csdn.net/Louis210/article/deJava中的List类的contains和indexOf方法的区别tails/117415546"
    # original_sentence = "Java中的List类的contains和indexOf方法的区别"
    # # baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&pn=100&rn=50"+"&oq="+original_sentence+"&ie=utf-8"
    # baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&rn=50" + "&oq=" + original_sentence + "&ie=utf-8"
    # # print(baidu_url)
    # # html = Pretreatment.get_raw_html(baidu_url)
    # # bf = BeautifulSoup(html, "html.parser")
    # # print(bf.prettify())
    # f = open("../src/text/html.txt")
    # html = f.read()
    # # print(html)
    # f.close()
    # # print(Pretreatment.get_next_baidu_url(html))
    #
    # bf = BeautifulSoup(html, "html.parser")
    # contents = bf.find_all("div", class_="c-container")
    # i = 1
    # article_urls = list()
    # red_strings = list()
    # print("下一页>>>>>>>>>>>" + Pretreatment.get_next_baidu_url(html))
    # print(len(contents))
    # for content in contents:
    #     # print("------------第{}个开始--------------".format(i))
    #     # print(content)
    #     # print("------------第{}个结束--------------".format(i))
    #     red_string = set()
    #     i += 1
    #     temp = dict()
    #     article_url = ""
    #     flag = False
    #     for child in content.children:
    #         if isinstance(child, bs4.element.NavigableString):
    #             continue
    #         for c in child.children:
    #             if c.name == "a" and c.parent.name == "h3":
    #                 article_url = c.attrs['href']
    #                 # print(">>>>>>>>>>>>", end="")
    #                 real_url = Pretreatment.get_real_url(article_url)
    #                 if real_url != "":
    #                     flag = True
    #                     article_urls.append(real_url)
    #                 # print()
    #     if flag:
    #         for child in content.descendants:
    #             if child.name == "em" and child.string != "":
    #                 red_string.add(child.string)
    #         red_strings.append(red_string)
    #
    # print(len(article_urls))
    # print(len(red_strings))
    # pprint(article_urls)
    # pprint(red_strings)

    # url = "https://blog.csdn.net/Louis210/article/details/117415546"
    # original_sentence = "Java中的List类的contains和indexOf方法的区别"
    # pprint(Pretreatment.get_related_paragraphs_and_sentences(original_sentence, paragraph_number=5, sentence_number=10,
    #                                                          url=url))

    # url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    # # url = "https://www.cnblogs.com/yuyueq/p/15119512.html"x
    # # url = "https://starlooo.github.io/2021/07/02/CaiKeng/"
    # # url = "https://www.jianshu.com/p/92373a603d42"
    # similarity = Pretreatment.split_txt(url)
    # i = 1
    # for code in similarity['codes']:
    #     codes = Pretreatment.clean_code(code)
    #     print("------第{}个代码段------".format(i))
    #     i += 1
    #     for line in codes:
    #         print("开始搜索：" + line)
    #         print(Pretreatment.get_related_codes(line, 5))
