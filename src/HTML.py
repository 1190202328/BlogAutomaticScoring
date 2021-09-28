import random
import bs4
import requests
import time
from bs4 import BeautifulSoup


def is_valid(proxy, url, verbose=True):
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
    # baidu_url = "http://baidu.com/s?wd=java&rn=50&oq=java&ie=utf-8"
    baidu_url = url
    retry_count = 3

    while retry_count > 0:
        try:
            if verbose:
                print("检测代理是否可用的第{}次尝试".format(4 - retry_count))
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
            return html
        except Exception:
            # print("ip第{}次检测出不行>>>".format(6 - retry_count), proxy)
            time.sleep(random.randrange(3, 6, 1))
            if verbose:
                print("失败")
            retry_count -= 1
            continue
    return ""


def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").json()


def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete?proxy={}".format(proxy))


def get_raw_html(url, verbose=True):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, compress',
        'Accept-Language': 'en-us;q=0.5,en;q=0.3',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
    }
    retry_count = 200
    proxy = get_proxy().get("proxy")
    html = ""
    while proxy and retry_count > 0:
        if verbose:
            print("<第{}次尝试>".format(21 - retry_count))
        html = is_valid(proxy, url, verbose=verbose)
        if html == "":
            if verbose:
                print("ip地址被删除>>>", proxy)
            delete_proxy(proxy)
            proxy = get_proxy().get("proxy")
            retry_count -= 1
        else:
            if verbose:
                print("获得的ip地址>>>", proxy)
            break
    if retry_count == 0 or not proxy:
        try:
            r = requests.get(url,
                             headers=headers,
                             timeout=5)
            time.sleep(random.randrange(30, 60, 1))
            if verbose:
                print("成功")
            return r.text
        except Exception:
            return ""
    return html


def get_raw_html_origin(url, verbose=False):
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
    # # proxy = "127.0.0.1:11000" # QuickQ
    # proxy = "127.0.0.1:7890"  # clashX
    # proxies = {
    #     'http': 'http://' + proxy,
    #     'https': 'https://' + proxy
    # }, proxies=proxies
    try:
        r = requests.get(url=url, headers=headers, timeout=5)
        r.raise_for_status()
        return r.text
    except Exception as e:
        if verbose:
            print(e.args)
        return ""


def get_real_url(url, verbose=False):
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
        r = requests.get(url=url, headers=headers, timeout=5)
        r.raise_for_status()
        return r.url
    except Exception as e:
        if verbose:
            print(e.args)
        return ""
    


if __name__ == '__main__':
    original_sentence = "java"
