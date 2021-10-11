import json
import random
import re
import bs4
import requests
import time
from bs4 import BeautifulSoup

from src.others import Global


def is_valid_baidu(proxy: str, url: str, verbose: bool = True) -> str:
    """
    检验baidu代理是否可用
    :param url: 检测目的百度url地址
    :param verbose: 是否繁杂输出
    :param proxy:代理ip
    :return:html文档
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
            if re.match(Global.not_find, bf.get_text(), flags=re.S):
                if verbose:
                    print("成功>>>百度搜不到这个关键字")
                return html
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
                    print("访问百度失败")
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
                print("ip是坏的，失败")
            retry_count -= 1
            continue
    return ""


def is_valid_github(proxy: str, url: str, verbose: bool = True) -> str:
    """
    检验github代理是否可用
    :param url: 检测目的github的url地址
    :param verbose: 是否繁杂输出
    :param proxy:代理ip
    :return:html文档
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
    retry_count = 3

    while retry_count > 0:
        try:
            if verbose:
                print("检测代理是否可用的第{}次尝试".format(4 - retry_count))
            r = requests.get(url,
                             headers=headers,
                             proxies={"http": "http://{}".format(proxy), "https": "https://{}".format(proxy)},
                             timeout=5)
            if r.status_code == 404 or r.status_code == 451:
                return 'not_find'
            if r.status_code == 403:
                time.sleep(random.randrange(60, 90))
                return 'not_find'
            r.raise_for_status()
            html = r.text
            if re.match('.+Not Found.+', html, flags=re.S) or re.match('.+Repository access blocked.+', html,
                                                                       flags=re.S):
                return 'not_find'
            api_result = json.loads(html)
            update_time = api_result['updated_at'][:10]
            if verbose:
                print(url, ' >>> ', update_time)
            return html
        except Exception as e:
            time.sleep(random.randrange(3, 6, 1))
            if verbose:
                print("github-api返回结果出错或ip是坏的>>>", url, e)
            retry_count -= 1
            continue
    return ""


def get_proxy():
    """
    从网页获得代理
    :return: 代理ip
    """
    return requests.get("http://127.0.0.1:5010/get/").json()


def delete_proxy(proxy):
    """
    从网页删除代理
    :param proxy: 代理ip
    :return: 无
    """
    requests.get("http://127.0.0.1:5010/delete?proxy={}".format(proxy))


def get_raw_html(url: str, is_valid=is_valid_baidu, verbose: bool = True) -> str:
    """
    获取html页面
    :param url: url地址
    :param is_valid: 检测是否成功的函数，默认为百度url
    :param verbose: 是否繁杂输出
    :return: html页面
    """
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, compress',
        'Accept-Language': 'en-us;q=0.5,en;q=0.3',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
    }
    retry_count = 25
    proxy = get_proxy().get("proxy")
    html = ""
    while proxy and retry_count > 0:
        if verbose:
            print("<第{}次尝试>".format(26 - retry_count))
        html = is_valid(proxy, url, verbose=verbose)
        if html == 'not_find':
            return ''
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
        except Exception as e:
            if verbose:
                print(e.args)
            return ""
    return html


def get_raw_html_origin(url: str, verbose: bool = False) -> str:
    """
    根据url获取html文档
    :param verbose: 是否繁杂输出
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
        r = requests.get(url=url, headers=headers, timeout=5)
        r.raise_for_status()
        return r.text
    except Exception as e:
        if verbose:
            print(e.args)
        return ""


def get_real_url(url: str, verbose: bool = False) -> str:
    """
    根据url获得真实的url地址
    :param verbose: 是否繁杂输出
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
    print(get_raw_html('https://api.github.com/repos/Bukkit/CraftBukkit', is_valid=is_valid_github))
    # print(get_raw_html('https://api.github.com/repositories/641377', is_valid=is_valid_github))
