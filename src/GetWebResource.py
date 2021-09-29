import re
from pprint import pprint
from typing import Union, Any, Optional

from bs4 import BeautifulSoup

from src import HTML, Global, Clean
from src.EDU import demo
from src.SeparateCode import SeparateCode


def split_txt(txt_url: str, EDU: bool = False, verbose: bool = True) -> Optional[
    dict[str, Union[Union[str, list[str], list[Union[str, Any]]], Any]]]:
    """
    根据url地址返回一个词典，词典中包含以下属性：1。head：标题；2。paragraphs：段落；3。sentences：句子；4。codes：代码；
    5。date：日期；6。text：全文（不含代码段）；
    :param verbose: 是否繁杂输出
    :param EDU: 是否采用EDU来划分句子
    :param txt_url: url地址
    :return: 词典，如果不满足目的url（1。csdn2。cnblogs3。github），则返回None
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

    is_illegal = False
    url = HTML.get_real_url(txt_url, verbose=verbose)
    html = HTML.get_raw_html_origin(url, verbose=verbose)
    if html == "":
        return None
    bf = BeautifulSoup(html, "html.parser")

    if re.match(Global.url_pattern['csdn'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="title-article")
        if content is None:
            if verbose:
                print("这个url标题有问题：" + txt_url)
            return None
        head = content.text.replace("\n", "")
        # text
        # text = bf.find("div", id="content_views").get_text()
        for child in bf.find("div", id="content_views"):
            if child.name != "pre":
                if child.string is not None:
                    text += child.string
                else:
                    text += child.get_text(separator="\n")
        # date
        update_date = bf.find("span", class_="time").text[0:10]
        # codes
        contents = bf.find_all("pre")
        for content in contents:
            codes.append(content.getText())
    if re.match(Global.url_pattern['cnblogs'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="postTitle")
        if content is None:
            content = bf.find("a", id="cb_post_title_url")
            if content is None:
                if verbose:
                    print("这个url标题有问题：" + txt_url)
                return None
        head = content.text.replace("\n", "")
        # text
        text = bf.find("div", id="cnblogs_post_body").getText()
        # for child in bf.find("div", id="cnblogs_post_body"):
        #     if child.name != "pre":
        #         if child.string is not None:
        #             text += child.string
        #         else:
        #             text += child.get_text(separator="\n")
        # date
        update_date = bf.find("span", id="post-date").text[0:10]
        # codes
        contents = bf.find_all("pre")
        for content in contents:
            codes.append(content.getText())
    if re.match(Global.url_pattern['github'], url):
        is_illegal = True
        # head
        content = bf.find("h1", class_="post-title")
        if content is None:
            content = bf.find("h1", class_="article-title sea-center")
            if content is None:
                content = bf.find("h1", class_="article-title")
                if content is None:
                    if verbose:
                        print("这个url标题有问题：" + txt_url)
                    return None
        head = content.text.replace("\n", "")
        # text
        text = bf.find("div", itemprop="articleBody").getText()
        # date
        update_date = bf.find("time").attrs['datetime'][0:10]
        # codes
        contents = bf.find_all("pre")
        digits = list()
        delete_codes = list()
        for content in contents:
            if re.match("\\d+", content.getText()):
                digits.append(content.getText())
                continue
            delete_codes.append(content.getText())
            raw_code = ""
            for child in content.children:
                if child.name == "span":
                    if child.string is not None:
                        raw_code += child.string + "\n"
            codes.append(raw_code)
        for digit in digits:
            start = text.find(digit)
            if start != -1:
                text = text[0:start] + text[start + len(digit):]
        for delete_code in delete_codes:
            start = text.find(delete_code)
            if start != -1:
                text = text[0:start] + text[start + len(delete_code):]

    if is_illegal:
        for code in codes:
            start = text.find(code)
            if start != -1:
                text = text[0:start] + text[start + len(code):]
        text = re.sub("(\\xa0)|(\\u200b)|(\\u2003)|(\\u3000)", "", text)
        text = re.sub("[\\t ]+", " ", text)
        text = re.sub("\n+", "\n", text)
        text = re.sub("(\n +)|( +\n)", "\n", text)
        to_search_code_text = Clean.clean_code_for_text(text)
        more_codes = SeparateCode.get_codes(to_search_code_text)
        if more_codes:
            for more_code in more_codes:
                start = text.find(more_code)
                if start != -1:
                    text = text[0:start] + text[start + len(more_code):]
            codes += more_codes
        text = re.sub("\n+", "\n", text)
        # paragraphs
        paragraphs = text.split("\n")
        lenth = 200
        for paragraph in paragraphs:
            paragraph = re.sub("\\s+", " ", paragraph)
            if paragraph != " " and len(paragraph) > 2:
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
        text = ""
        for clean_paragraph in clean_paragraphs:
            text += clean_paragraph + "\n"
        # sentences
        if EDU:
            if clean_text_for_EDU_element != "":
                clean_text_for_EDU.append(clean_text_for_EDU_element)
            total_num = len(clean_text_for_EDU)
            j = 1
            for text in clean_text_for_EDU:
                if verbose:
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
        if head == '' or re.match('\\s+', head):
            head = sentences[0]
        result['head'] = head
        result['paragraphs'] = clean_paragraphs
        result['sentences'] = sentences
        result['codes'] = codes
        result['date'] = update_date
        result['text'] = text
        return result
    else:
        return None


def get_urls(main_url: str, verbose: bool = True) -> []:
    """
    根据学生主页面获取所有博客的url地址
    :param verbose: 是否繁杂输出
    :param main_url: 主页面地址，包括（1。csdn2。cnblogs3。github4。简书）
    :return: 所有博客的ulr地址
    """
    html = HTML.get_raw_html_origin(main_url, verbose=verbose)
    bf = BeautifulSoup(html, "html.parser")
    urls = set()
    contents = bf.find_all("a")
    if re.match(Global.pattern_csdn_main, main_url):
        for content in contents:
            if content.get("href") is not None and re.match(".*/article/details.*", content.get("href")):
                if re.match(".*#comments|.*blogdevteam.*", content.get("href")):
                    continue
                urls.add(content.get("href"))
    if re.match(Global.pattern_cnblogs_main, main_url):
        for content in contents:
            if content.get("href") is not None and re.match(Global.pattern_cnblogs, content.get("href")):
                urls.add(content.get("href"))
    if re.match(Global.pattern_github_main, main_url):
        for content in contents:
            if content.get("href") is not None and re.match("/\\d{4}/\\d{2}/\\d{2}/.+/", content.get("href")) \
                    and not re.match("/\\d{4}/\\d{2}/\\d{2}/.+/#more", content.get("href")):
                urls.add(main_url + content.get("href"))
    return list(urls)


def get_main_url(url: str) -> str:
    """
    根据url地址返回主页的url地址
    :param url: url地址（1。csdn2。cnblogs3。github4。简书）
    :return: 主页的URL地址，如果找不到则返回""
    """
    main_url = None
    if re.match(Global.pattern_csdn_main, url):
        temps = url.split("/")
        main_url = "https://blog.csdn.net/" + temps[3]
    if re.match(Global.pattern_cnblogs_main, url):
        temps = url.split("/")
        main_url = "https://www.cnblogs.com/" + temps[3]
    if re.match(Global.pattern_github_main, url):
        temps = url.split("/")
        main_url = "https://" + temps[2]
    return main_url


if __name__ == '__main__':
    total_urls_get = []
    with open('../src/text/所有文章的url.txt', mode='r') as f:
        for line in f.readlines():
            total_urls_get.append(line[:-1])
    print(len(total_urls_get))

    start = 1315
    total_urls_get = total_urls_get[start:]
    i = start
    for url in total_urls_get:
        result = split_txt(url)
        if result is None:
            print('\033[1;31;40m这个url有问题>>> '+url+'\033[0m')
        else:
            update_date = result.get('date')
            if not re.match('\\d{4}-\\d{2}-\\d{2}', update_date):
                print('这个日期格式有错误>>>', i, ' ', update_date)
                break
            print('<{}> '.format(i), update_date)
        i += 1

