import re

import requests
from bs4 import BeautifulSoup


class BlogAutomaticScoring:
    """
    读取学生自动打分的工具类
    """

    @staticmethod
    def get_text(url):
        """
        根据url地址获取页面内容
        :param url: 页面所在URL地址
        :return: 页面文档
        """
        req = requests.get(url=url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        text = ""
        pattern = "</p>|<p(.*)>|\\n"
        bf = BeautifulSoup(html)
        contents = bf.find_all("div", class_="blog-content-box")
        for content in contents:
            if content.class_ == ("blog-tags-box" or "bar-content"):
                continue
            if (content.tag == "p" or "h1" or "ul") and not content.text.isspace():
                text += re.sub(pattern, "\n", content.text, count=0, flags=0) + "\n"
        text = re.sub("[ \t]", "", text)
        text = re.sub("\n+", "\n", text)
        return text

    @staticmethod
    def get_urls(main_url):
        """
        根据学生主页面获取所有博客的url地址
        :param main_url: 主页面地址
        :return: 所有博客的ulr地址
        """
        req = requests.get(url=main_url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html)
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
        bf = BeautifulSoup(html)
        contents = bf.find_all("a", href=True)
        for content in contents:
            if re.match("https://blog.csdn.net/\\w+", content.get("href")):
                return content.get("href")
        return None
