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
        :param url: URL地址
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
