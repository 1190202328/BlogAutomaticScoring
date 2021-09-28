import json
import keyword
import re
from pprint import pprint

import requests

from src import Clean

"""
数据集的代码涵盖1.python 2.java 3.C 三个主流语言
非代码:1.中文句子2.英文句子
标签: 0:代码 1:非代码
"""


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


def get_related_codes(code, number, low_limit=7, high_limit=30):
    """
    根据code获取相关的code
    :param high_limit: 每行代码的最长长度，大于该长度的代码行将会被过滤
    :param low_limit: 每行代码的最短长度，小于该长度的代码行将会被过滤
    :param code: 一行源代码
    :param number: 需要获取相关code的数量，最多100行相关的code
    :return: 相关code的列表(code中不含中文注释)
    """
    print("开始搜索：" + code)
    pn = 0
    count = 0
    related_codes = []
    api = "https://searchcode.com/api/codesearch_I/?q=" + code + "&p=" + pn.__str__() + "&per_page=100"
    while count < number:
        text = get_raw_html(api)
        if text == "":
            print("ip被封了！")
            return related_codes
        api_result = json.loads(text)
        # pprint(api_result)
        pn = api_result.get('nextpage')
        if pn > 100:
            break
        api = "https://searchcode.com/api/codesearch_I/?q=" + code + "&p=" + pn.__str__() + "&per_page=100"
        results = api_result.get('results')
        if results is None:
            print("找不到该代码的相似代码")
            print("url地址>>>>>>>>>>>>" + api)
            return related_codes
        for result in results:
            lines = list(result['lines'].values())
            # print(lines)
            clean_line = Clean.clean_code_line(lines[1], low_limit, high_limit)
            if clean_line != "" and not re.match("\\s+", clean_line):
                related_codes.append(clean_line)
                count += 1
                print("找到第{}个相关代码".format(count))
                if count >= number:
                    return related_codes
    return related_codes


def write_code(path, file, codes):
    f = open(path + file, "a")
    for code in codes:
        f.write(code)
        f.write("\n")
    f.close()


def get_key_words(path, file):
    key_words = list()
    f = open(path + file, "r")
    for line in f.readlines():
        key_words.append(line)
    f.close()
    return key_words


if __name__ == '__main__':
    # write_code("","python关键字.txt",keyword.kwlist)
    path = "../src/text/"
    key_words = set(get_key_words(path, "不重复关键字.txt"))
    i = 1
    for key_word in key_words:
        print("第{}个(共{}个)".format(i, len(key_words)), end="")
        i += 1
        write_code(path, "扩大的代码.txt", get_related_codes(key_word.replace("\n", ""), 200, high_limit=50))
