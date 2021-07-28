from pprint import pprint

from bs4 import BeautifulSoup
import requests
import re
from datetime import date
from baiduspider import BaiduSpider

from src.demo import get_EDUs


class Pretreatment:
    """
    预处理类，负责从网页抓取文档，预处理文档。
    """

    @staticmethod
    def split_txt(url):
        """
        根据url地址返回一个词典，词典中包含以下属性：1。head：标题；2。paragraphs：段落；3。sentences：句子；4。codes：代码；5。date：日期；
        :param url: url地址
        :return: 词典
        """
        result = dict()
        head = ""
        paragraphs = list()
        sentences = list()
        codes = list()
        update_date = ""

        req = requests.get(url=url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("h1", class_="title-article")

        # print(bf.get_text())

        head = contents[0].text
        main_content_html = bf.find("div", id="content_views")
        for child in main_content_html.children:
            if child.name == "ul":
                for string in child.strings:
                    paragraphs.append(string)
            elif child.name == "pre":
                codes.append(child.string)
            elif child.string is None:
                continue
            else:
                if child.string != '\n':
                    paragraphs.append(child.string)
        contents = bf.find_all("span", class_="time")
        update_date = date.fromisoformat(contents[0].text[0:10])
        result['head'] = head
        result['paragraphs'] = paragraphs
        result['sentences'] = sentences
        result['codes'] = codes
        result['date'] = update_date
        text = "".join(paragraphs)
        result['text'] = text
        return result

    @staticmethod
    def get_all_texts(students):
        """
        获得所有博客
        :param students: 学生列表
        :return: 文章列表，一个元素为一篇文章
        """
        texts = list()
        for student in students:
            print(student)
            if student.url is None or (not re.match(".*csdn.*", student.url)):
                continue
            urls = Pretreatment.get_urls(Pretreatment.get_main_url(student.url))
            for url in urls:
                txt, _ = Pretreatment.get_text(url)
                txt = re.sub("(\\s+)|(\\.{3,})|(—+)", " ", txt)
                texts.append(txt)
        return texts

    @staticmethod
    def get_related_txt(txt_head, number):
        """
        根据标题在百度搜索相关文章，取出前number篇文章的url地址
        :param number: 需要相关文章的篇数
        :param txt_head: 文章标题
        :return: number篇文章的url的列表
        """
        total_urls = list()
        total_titles = list()
        count = 0
        pn = 1
        while True:
            results = BaiduSpider().search_web(txt_head, pn=pn, exclude=['all']).get('results')
            print("pn = {}".format(pn))
            pn += 1
            for result in results:
                print(result)
                if count >= number:
                    break
                if result.get('title') is None:
                    continue
                # if re.match(".*CSDN博客.*", result.get('title')):
                title = result.get('title').split("_")
                if len(title) == 1:
                    title = result.get('title').split("-")
                if (result.get('url') in total_urls) or (title in total_titles):
                    continue
                total_titles.append(title[0])
                total_urls.append(result.get('url'))
                print("count = {}".format(count))
                count += 1
            if count >= number:
                break
        return total_urls, total_titles

    @staticmethod
    def get_urls(main_url):
        """
        根据学生主页面获取所有博客的url地址
        :param main_url: 主页面地址
        :return: 所有博客的ulr地址
        """
        req = requests.get(url=main_url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("a", href=True)
        urls = set()
        for content in contents:
            if re.match(".*/article/details.*", content.get("href")):
                if re.match(".*#comments|.*blogdevteam.*", content.get("href")):
                    continue
                urls.add(content.get("href"))
        return urls

    @staticmethod
    def get_main_url(url):
        """
        根据url地址返回主页的url地址
        :param url: 任意url地址
        :return: 主页的URL地址，如果找不到则返回None
        """
        req = requests.get(url=url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("a", href=True)
        for content in contents:
            if re.match("https://blog.csdn.net/\\w+", content.get("href")):
                return content.get("href")
        return None

    @staticmethod
    def get_related_sentences(original_sentence, number):
        """
        在百度上获取相关的句子
        :param original_sentence: 源句子
        :param number: 获取数量
        :return: 相关句子的列表
        """
        sentences = list()
        urls = list()
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, compress',
            'Accept-Language': 'en-us;q=0.5,en;q=0.3',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
        }
        baidu_url = 'http://www.baidu.com'
        url = 'http://www.baidu.com.cn/s?wd=' + original_sentence + '&cl=3'
        html = requests.get(url=url, headers=headers).text
        print(html)
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find("div", id="content_left")
        for content in contents:
            print(content)
            # if child.class_ == "result c-container new-pmd":
            #     for ch in child.chilren:
            #         if ch.name == "h3":
            #             for c in ch.children:
            #                 urls.append(c.get_href)
        print(urls)
        return sentences


if __name__ == '__main__':
    # result = Pretreatment.split_txt("https://blog.csdn.net/Louis210/article/details/117415546")
    # print(result['head'])
    # for paragraph in result['paragraphs']:
    #     print(paragraph)
    # # print(result['sentences'])
    # for code in result['codes']:
    #     print(code)
    # print(result['date'])

    # original_sentence = "java的contains"
    # number = 1
    # sentences = Pretreatment.get_related_sentences(original_sentence, number)
    # print(sentences)

    result = Pretreatment.split_txt("https://blog.csdn.net/Louis210/article/details/117415546")
    text = "据统计，这些城市去年完成国内生产总值一百九十多亿元，比开放前的一九九一年增长九成多。国务院于一九九二年先后批准了黑河、凭祥、珲春、伊宁、瑞丽等十四个边境城市为对外开放城市，同时还批准这些城市设立十四个边境经济合作区。三年多来，这些城市社会经济发展迅速，地方经济实力明显增强；经济年平均增长百分之十七，高于全国年平均增长速度。以下是来测试测试这个代码的程序。 "
    text = result['text']
    print(text)
    EDUs = get_EDUs(text)
    pprint(EDUs)
