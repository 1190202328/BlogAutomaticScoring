from pprint import pprint
from bs4 import BeautifulSoup
import requests
import re
from datetime import date
from baiduspider import BaiduSpider

from src.BlogAutomaticScoring import BlogAutomaticScoring


class OutOfPageLimitError(RuntimeError):
    def __init__(self, message="超过搜索的最大页码限制！"):
        self.message = message


class Pretreatment:
    """
    预处理类，负责从网页抓取文档，预处理文档。
    """

    @staticmethod
    def split_txt(url):
        """
        根据url地址返回一个词典，词典中包含以下属性：1。head：标题；2。paragraphs：段落；3。sentences：句子；4。codes：代码；5。date：日期；6。text全文（不含代码段）
        :param url: url地址
        :return: 词典
        """
        result = dict()
        paragraphs = list()
        sentences = list()
        codes = list()

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
        for paragraph in paragraphs:
            sentences += re.split("[,.，。]", paragraph)
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
    def get_related_head(txt_head, number, page_limit=30):
        """
        根据标题在百度搜索相关文章，取出前number篇文章的标题
        :param page_limit: 最大页码数
        :param number: 需要相关文章的篇数
        :param txt_head: 文章标题
        :return: number篇文章的标题
        """
        total_urls = list()
        total_titles = list()
        count = 0
        pn = 1
        while True:
            results = BaiduSpider().search_web(txt_head, pn=pn, exclude=['all']).get('results')
            print("pn = {}".format(pn))
            pn += 1
            if pn > page_limit:
                return list()
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
        return total_titles

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
    def get_related_urls(query, number, page_limit=30):
        """
        根据query在百度搜索，取出前number篇csdn文章的url地址
        :param number: 需要相关文章的篇数
        :param query: 搜索字符串
        :return: number个csdn的url地址
        """
        total_urls = list()
        count = 0
        pn = 1
        while True:
            results = BaiduSpider().search_web(query, pn=pn, exclude=['all']).get('results')
            print("pn = {}".format(pn))
            pn += 1
            if pn > page_limit:
                return list()
            for result in results:
                if count >= number:
                    break
                if result.get('url') is None:
                    continue
                real_url = requests.get(result.get('url'))
                patten = "https://blog\\.csdn\\.net/.+"
                if re.match(patten, real_url.url):
                    total_urls.append(real_url.url)
                    count += 1
                    print("count = {}".format(count))
            if count >= number:
                break
        return total_urls

    @staticmethod
    def get_related_sentences(original_sentence, number=5, page_limit=30):
        """
        在百度上获取相关的句子(目前仅限于csdn博客)
        :param original_sentence: 源句子
        :param number: 获取的数量
        :return: 相关句子的列表
        """
        i = 0
        pn = 0
        count = 0
        related_sentences = set()
        while True:
            if count >= number:
                break
            url = 'http://www.baidu.com.cn/s?wd=' + original_sentence + '&cl=3' + "&pn=" + str(pn * 10)
            print("第{}页".format(pn))
            pn += 1
            if pn > page_limit:
                return list()
            html = Pretreatment.get_raw_html(url)
            bf = BeautifulSoup(html, "html.parser")
            contents = bf.find_all("div", class_="result c-container new-pmd")
            urls = list()
            for content in contents:
                temp = dict()
                url = ""
                red_strings = set()
                for child in content.children:
                    for c in child.children:
                        if c.name == "a" and c.parent.name == "h3":
                            url = c.attrs['href']
                real_url = requests.get(url)
                patten = "https://blog\\.csdn\\.net/.+"
                if not re.match(patten, real_url.url):
                    continue
                for child in content.descendants:
                    if child.name == "em":
                        red_strings.add(child.string)
                temp["url"] = real_url.url
                temp["red_strings"] = red_strings
                urls.append(temp)
                count += 1
                if count >= number:
                    break
            print("url共有{}个".format(len(urls)))
            print(urls)

            for dictionary in urls:
                print("第{}篇文章".format(i + 1))
                print("句子如下：")
                i += 1
                url = dictionary['url']
                result = Pretreatment.split_txt(url)
                sentences = result['sentences']
                for sentence in sentences:
                    for substring in dictionary['red_strings']:
                        if sentence.find(substring) != -1:
                            related_sentences.add(sentence)
                pprint(related_sentences)
        pprint(related_sentences)
        return list(related_sentences)

    @staticmethod
    def get_related_paragraphs(original_sentence, number=5, page_limit=30):
        """
        在百度上获取相关的段落(目前仅限于csdn博客)
        :param original_sentence: 需要查询的句子
        :param number: 获取的数量
        :return: 相关段落的列表
        """
        i = 0
        pn = 0
        count = 0
        related_paragraphs = set()
        while True:
            if count >= number:
                break
            url = 'http://www.baidu.com.cn/s?wd=' + original_sentence + '&cl=3' + "&pn=" + str(pn * 10)
            print("第{}页".format(pn))
            pn += 1
            if pn > page_limit:
                return list()
            html = Pretreatment.get_raw_html(url)
            bf = BeautifulSoup(html, "html.parser")
            contents = bf.find_all("div", class_="result c-container new-pmd")
            urls = list()
            for content in contents:
                temp = dict()
                url = ""
                red_strings = set()
                for child in content.children:
                    for c in child.children:
                        if c.name == "a" and c.parent.name == "h3":
                            url = c.attrs['href']
                real_url = requests.get(url)
                patten = "https://blog\\.csdn\\.net/.+"
                if not re.match(patten, real_url.url):
                    continue
                for child in content.descendants:
                    if child.name == "em":
                        red_strings.add(child.string)
                temp["url"] = real_url.url
                temp["red_strings"] = red_strings
                urls.append(temp)
                count += 1
                if count >= number:
                    break
            print("url共有{}个".format(len(urls)))
            print(urls)

            for dictionary in urls:
                print("第{}篇文章".format(i + 1))
                print("句子如下：")
                i += 1
                url = dictionary['url']
                result = Pretreatment.split_txt(url)
                paragraphs = result['paragraphs']
                for paragraph in paragraphs:
                    for substring in dictionary['red_strings']:
                        if paragraph.find(substring) != -1:
                            related_paragraphs.add(paragraph)
                pprint(related_paragraphs)
        return list(related_paragraphs)

    @staticmethod
    def get_raw_html(url):
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
        except requests.ConnectionError as e:
            print(e.args)


if __name__ == '__main__':
    original_sentence = "有一次使用到了contains和indexOf方法"
    Pretreatment.get_related_sentences(original_sentence)
