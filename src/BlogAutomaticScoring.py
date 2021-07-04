import re

import requests
from bs4 import BeautifulSoup


class BlogAutomaticScoring:
    """
    读取学生自动打分的工具类
    """

    @staticmethod
    def get_text(url):
        req = requests.get(url=url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html)
        head = bf.find_all("title", style="text-indent:33px;")
        p = bf.find_all("p")
        text = ""
        pattern = "</p>|<p(.*)>|\\n"
        for i in range(len(p)):
            text += re.sub(pattern, "", p[i].text, count=0, flags=0)
        return text
