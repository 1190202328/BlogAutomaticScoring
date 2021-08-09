from pprint import pprint
from bs4 import BeautifulSoup
import requests
import re
from datetime import date
from baiduspider import BaiduSpider
from src import demo


class Pretreatment:
    """
    预处理类，负责从网页抓取文档，预处理文档。
    """

    @staticmethod
    def split_txt(txt_url, EDU=False):
        """
        根据csdn的url地址返回一个词典，词典中包含以下属性：1。head：标题；2。paragraphs：段落；3。sentences：句子；4。codes：代码；5。date：日期；6。text全文（不含代码段）
        :param txt_url: url地址
        :return: 词典，如果不满足目的url（1。csdn），则返回None
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
        patten_csdn = "https://blog\\.csdn\\.net/.+"
        patten_cnblogs = "https://www\\.cnblogs\\.com/.+"
        patten_github = "https://.+\\.github\\.io/.+"
        patten_jianshu = "https://www\\.jianshu\\.com/.+"

        is_illegal = False
        url = Pretreatment.get_real_url(txt_url)
        html = Pretreatment.get_raw_html(url)
        bf = BeautifulSoup(html, "html.parser")

        if re.match(patten_csdn, url):
            is_illegal = True
            # head
            content = bf.find("h1", class_="title-article")
            if content is None:
                print("这个url标题有问题：" + txt_url)
            head = content.text.replace("\n", "")
            # text
            text = bf.find("div", id="content_views").getText()
            # date
            update_date = date.fromisoformat(bf.find("span", class_="time").text[0:10])
            # codes
            contents = bf.find_all("pre")
            for content in contents:
                codes.append(content.getText())
        if re.match(patten_cnblogs, url):
            is_illegal = True
            # head
            content = bf.find("h1", class_="postTitle")
            if content is None:
                print("这个url标题有问题：" + txt_url)
            head = content.text.replace("\n", "")
            # text
            text = bf.find("div", id="cnblogs_post_body").getText()
            # date
            update_date = date.fromisoformat(bf.find("span", id="post-date").text[0:10])
            # codes
            contents = bf.find_all("pre")
            for content in contents:
                codes.append(content.getText())
        if re.match(patten_github, url):
            is_illegal = True
            # head
            content = bf.find("h1", class_="post-title")
            if content is None:
                content = bf.find("h1", class_="article-title sea-center")
                if content is None:
                    print("这个url标题有问题：" + txt_url)
            head = content.text.replace("\n", "")
            # text
            text = bf.find("div", itemprop="articleBody").getText()
            # date
            update_date = date.fromisoformat(bf.find("time").attrs['datetime'][0:10])
            # codes
            contents = bf.find_all("pre")
            digits = list()
            for content in contents:
                if re.match("\\d+", content.getText()):
                    digits.append(content.getText())
                    continue
                codes.append(content.getText())
            for digit in digits:
                start = text.find(digit)
                if start != -1:
                    text = text[0:start] + text[start + len(digit):]
        if re.match(patten_jianshu, url):
            is_illegal = True
            # head
            content = bf.find("h1", class_="_1RuRku")
            if content is None:
                print("这个url标题有问题：" + txt_url)
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
            text = re.sub("\n+", "\n", text)
            text = re.sub("(\\xa0)|(\\u200b)", "", text)
            # paragraphs
            paragraphs = text.split("\n")
            lenth = 200
            for paragraph in paragraphs:
                paragraph = re.sub("\\s+", "", paragraph)
                if paragraph != "":
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
    def get_related_head(txt_head, number, page_limit=10, url=""):
        """
        根据标题在百度搜索相关文章，取出前number篇文章的标题
        :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
        :param page_limit: 最大页码数
        :param number: 需要相关文章的篇数
        :param txt_head: 文章标题
        :return: number篇文章的标题,find
        """
        total_urls = list()
        total_titles = list()
        count = 0
        pn = 1
        find = False
        while True:
            results = BaiduSpider().search_web(txt_head, pn=pn, exclude=['all']).get('results')
            print("pn = {}".format(pn))
            pn += 1
            if pn > page_limit:
                return total_titles, find
            for result in results:
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
                print("count = {}".format(count))
                count += 1
            if count >= number:
                break
        return total_titles, find

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
    def get_related_urls(query, number, page_limit=10, url=""):
        """
        根据query在百度搜索，取出前number篇csdn文章的url地址
        :param page_limit: 百度搜索的最大页码限制
        :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
        :param number: 需要相关文章的篇数
        :param query: 搜索字符串
        :return: number个csdn的url地址,find
        """
        total_urls = list()
        find = False
        count = 0
        pn = 1
        while True:
            results = BaiduSpider().search_web(query, pn=pn, exclude=['all']).get('results')
            print("pn = {}".format(pn))
            pn += 1
            if pn > page_limit:
                return total_urls, find
            for result in results:
                if count >= number:
                    break
                if result.get('url') is None:
                    continue
                real_url = Pretreatment.get_real_url(result.get('url'))
                patten = "https://blog\\.csdn\\.net/.+"
                if real_url in total_urls:
                    continue
                if url.find(real_url) != -1:
                    find = True
                    continue
                if re.match(patten, real_url):
                    total_urls.append(real_url)
                    count += 1
                    print("count = {}".format(count))
            if count >= number:
                break
        return total_urls, find

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
            r = requests.get(url=url, headers=headers, timeout=30)
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
            r = requests.get(url=url, headers=headers, timeout=30)
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
        urls = bf.find_all("a")
        for url in urls:
            if url.string == "下一页 >":
                return "https://www.baidu.com" + url.attrs['href']
        return ""

    @staticmethod
    def get_related_paragraphs_and_sentences(original_sentence, paragraph_number=5, sentence_number=5, page_limit=10,
                                             url=""):
        """
        在百度上获取相关的句子(目前仅限于csdn博客)
        :param sentence_number: 需要搜索的相关句子的文章篇数
        :param paragraph_number: 需要搜索的相关段落的文章篇数
        :param url: 可选，源文章第url地址，输入之后不会重复找到该文章，如果找到，则返回find=True
        :param page_limit: 百度搜索的页码限制
        :param original_sentence: 源句子
        :return: 相关段落的列表，相关句子的列表，find
        """
        pn = 0
        count = 0
        total_urls = list()
        find = False
        related_sentences = list()
        related_paragraphs = list()
        baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&oq=" + original_sentence + "&ie=utf-8"
        number = max(paragraph_number, sentence_number)
        urls = list()
        while True:
            article_urls = list()
            if count >= number:
                break
            html = Pretreatment.get_raw_html(baidu_url)
            baidu_url = Pretreatment.get_next_baidu_url(html)
            print("第{}页".format(pn))
            pn += 1
            if baidu_url == "":
                baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&pn=" + str(
                    pn * 10) + "&oq=" + original_sentence + "&ie=utf-8"
            if pn > page_limit:
                break
            bf = BeautifulSoup(html, "html.parser")
            contents = bf.find_all("div", class_="result c-container new-pmd")
            for content in contents:
                temp = dict()
                article_url = ""
                red_strings = set()
                for child in content.children:
                    for c in child.children:
                        if c.name == "a" and c.parent.name == "h3":
                            article_url = c.attrs['href']
                real_url = Pretreatment.get_real_url(article_url)
                article_urls.append(real_url)
                if real_url in total_urls:
                    continue
                if url.find(real_url) != -1:
                    find = True
                    continue
                patten = "https://blog\\.csdn\\.net/.+"
                if not re.match(patten, real_url):
                    continue
                for child in content.descendants:
                    if child.name == "em" and child.string != "":
                        red_strings.add(child.string)
                temp["url"] = real_url
                temp["red_strings"] = red_strings
                urls.append(temp)
                total_urls.append(real_url)
                count += 1
                if count >= number:
                    break
            print("url共有{}个".format(len(article_urls)))
        print(urls)
        i = 0
        for num in range(min(paragraph_number, len(urls))):
            print("第{}篇文章".format(i + 1))
            print("段落如下：")
            i += 1
            article_url = urls[num]['url']
            article_paragraphs = list()
            result = Pretreatment.split_txt(article_url)
            paragraphs = result['paragraphs']
            for paragraph in paragraphs:
                for substring in urls[num]['red_strings']:
                    if paragraph != "" and paragraph.find(substring) != -1:
                        related_paragraphs.append(paragraph)
                        article_paragraphs.append(paragraph)
                        break
            pprint(article_paragraphs)
        i = 0
        for num in range(min(sentence_number, len(urls))):
            print("第{}篇文章".format(i + 1))
            print("句子如下：")
            article_sentences = list()
            i += 1
            article_url = urls[num]['url']
            result = Pretreatment.split_txt(article_url)
            sentences = result['sentences']
            for sentence in sentences:
                for substring in urls[num]['red_strings']:
                    if sentence != "" and sentence.find(substring) != -1:
                        related_sentences.append(sentence)
                        article_sentences.append(sentence)
                        break
            pprint(article_sentences)
        return related_paragraphs, related_sentences, find


if __name__ == '__main__':
    # original_sentence = "有一次使用到了contains和indexOf方法"
    # Pretreatment.get_related_sentences(original_sentence)

    url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    url = "https://www.cnblogs.com/yuyueq/p/15119512.html"
    url = "https://starlooo.github.io/2021/07/02/CaiKeng/"
    url = "https://www.jianshu.com/p/92373a603d42"

    result = Pretreatment.split_txt(url)
    print("---------head---------")
    print(result['head'])
    print("---------text---------")
    print(result['text'])
    print("---------paragraphs---------")
    pprint(result['paragraphs'])
    print("---------sentences---------")
    pprint(result['sentences'])
    print("---------code---------")
    i = 1
    for code in result['codes']:
        print("-----------code{}-----------".format(i))
        i += 1
        print(code)
    print("---------date---------")
    print(result['date'])

    # result = Pretreatment.split_txt(url, EDU=True)
    # print("---------EDU-sentences--------")
    # pprint(result['sentences'])

    # original_sentence = "Java中的List类的contains和indexOf方法的区别"
    # url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    # baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&oq=" + original_sentence + "&ie=utf-8"
    # html = Pretreatment.get_raw_html(baidu_url)
    # # print(Pretreatment.get_next_baidu_url(html))
    # for i in range(15):
    #     print("第{}个url".format(i+1))
    #     print(baidu_url)
    #     html = Pretreatment.get_raw_html(baidu_url)
    #     baidu_url = Pretreatment.get_next_baidu_url(html)

    # print(Pretreatment.get_related_paragraphs_and_sentences(original_sentence, paragraph_number=5, sentence_number=3,
    #                                                         url=url))
