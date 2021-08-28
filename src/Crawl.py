import random
import bs4
import requests
import time
from bs4 import BeautifulSoup


class Crawl:
    """
    用于获取html页面的工具类
    """

    @staticmethod
    def is_valid(proxy, verbose=True):
        """
        检验代理是否可用
        :param proxy:
        :return:
        """
        if verbose:
            print("正在检测的ip地址>>>", proxy)
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, compress',
            'Accept-Language': 'en-us;q=0.5,en;q=0.3',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
        }
        baidu_url = "http://baidu.com/s?wd=java&rn=50&oq=java&ie=utf-8"
        retry_count = 2

        while retry_count > 0:
            try:
                if verbose:
                    print("检测代理是否可用的第{}次尝试".format(3 - retry_count))
                r = requests.get(baidu_url,
                                 headers=headers,
                                 proxies={"http": "http://{}".format(proxy), "https": "https://{}".format(proxy)},
                                 timeout=5)
                r.raise_for_status()
                article_urls = list()
                html = r.text
                bf = BeautifulSoup(html, "html.parser")
                contents = bf.find_all("div", class_="c-container")
                for content in contents:
                    for child in content.children:
                        if isinstance(child, bs4.element.NavigableString):
                            continue
                        for c in child.children:
                            if c.name == "a" and c.parent.name == "h3":
                                article_urls.append(c.attrs['href'])
                                break
                if len(article_urls) == 0:
                    # print("ip第{}次检测出不行>>>".format(6 - retry_count), proxy)
                    if verbose:
                        print("失败")
                    retry_count -= 1
                    time.sleep(random.randrange(3, 6, 1))
                    continue
                if verbose:
                    print("成功>>>url有{}个".format(len(article_urls)))
                return True
            except Exception:
                # print("ip第{}次检测出不行>>>".format(6 - retry_count), proxy)
                time.sleep(random.randrange(3, 6, 1))
                if verbose:
                    print("失败")
                retry_count -= 1
                continue
        return False

    @staticmethod
    def get_proxy():
        return requests.get("http://127.0.0.1:5010/get/").json()

    @staticmethod
    def delete_proxy(proxy):
        requests.get("http://127.0.0.1:5010/delete?proxy={}".format(proxy))

    @staticmethod
    def get_raw_html(url, verbose=True):
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, compress',
            'Accept-Language': 'en-us;q=0.5,en;q=0.3',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
        }
        retry_count = 5
        proxy = Crawl.get_proxy().get("proxy")
        while proxy and not Crawl.is_valid(proxy, verbose=verbose):
            if verbose:
                print("ip地址被删除>>>", proxy)
            Crawl.delete_proxy(proxy)
            proxy = Crawl.get_proxy().get("proxy")
        if verbose:
            print("获得的ip地址>>>", proxy)
        time.sleep(random.randrange(20, 30, 1))
        while retry_count > 0:
            try:
                if verbose:
                    print("第{}次尝试".format(6 - retry_count))
                if retry_count == 1 or not proxy:
                    r = requests.get(url,
                                     headers=headers,
                                     timeout=5)
                    time.sleep(random.randrange(10, 30, 1))
                else:
                    r = requests.get(url,
                                     headers=headers,
                                     proxies={"http": "http://{}".format(proxy), "https": "https://{}".format(proxy)},
                                     timeout=5)
                r.raise_for_status()
                if verbose:
                    print("成功")
                return r.text
            except Exception:
                if verbose:
                    print("失败")
                retry_count -= 1
                while proxy and not Crawl.is_valid(proxy, verbose=verbose):
                    if verbose:
                        print("ip地址被删除>>>", proxy)
                    Crawl.delete_proxy(proxy)
                    proxy = Crawl.get_proxy().get("proxy")
                time.sleep(random.randrange(20, 30, 1))
                continue
        if verbose:
            print("无法访问")
        return ""

    @staticmethod
    def get_related_paragraphs_and_sentences(original_sentence):
        baidu_url = 'http://baidu.com/s?wd=' + original_sentence + "&rn=50" + "&oq=" + original_sentence + "&ie=utf-8"
        article_urls = list()
        html = Crawl.get_raw_html(baidu_url)
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("div", class_="c-container")
        for content in contents:
            for child in content.children:
                if isinstance(child, bs4.element.NavigableString):
                    continue
                for c in child.children:
                    if c.name == "a" and c.parent.name == "h3":
                        article_urls.append(c.attrs['href'])
                        break
        if len(article_urls) == 0:
            print("此url暂时不能访问>>>")
            time.sleep(random.randrange(100, 200, 1))
        else:
            print("url共有{}".format(len(article_urls)))


if __name__ == '__main__':
    original_sentence = "java"
    Crawl.get_related_paragraphs_and_sentences(original_sentence)
