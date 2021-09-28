import re
import jieba


def clean_code_for_text(text: str) -> str:
    """
    获得不含注释的text
    目前注释只支持
    单行注释：//和#
    多行注释：/* */和三个'和三个"
    :param text: 源代码段
    :return: 不含注释的干净的text
    """
    codes = list()
    text = re.sub("/(\\*).*?(\\*)/", "", text, flags=re.S)
    text = re.sub("'''.*?'''", "", text, flags=re.S)
    text = re.sub('""".*?"""', "", text, flags=re.S)
    lines = text.split("\n")
    for line in lines:
        java_start = line.find("//")
        python_start = line.find("#")
        if java_start != -1:
            line = line[0:java_start]
        if python_start != -1:
            line = line[0:python_start]
        codes.append(line)
    return "\n".join(codes)


def clean_code(code: str, limit=7) -> []:
    """
    获得干净的codes列表（每一个元素为一行代码）（不含\t，\n，连续2个以上空格，注释，不含import,include等）
    目前注释只支持
    单行注释：//和#
    多行注释：/* */和三个'和三个"
    :param limit: 每行代码的最短长度，小于该长度的代码行将会被过滤
    :param code: 源代码段
    :return: codes列表
    """
    blank_pattern = "[\\t\\n]+"
    codes = list()
    code = re.sub("/(\\*).*?(\\*)/", "", code, flags=re.S)
    code = re.sub("'''.*?'''", "", code, flags=re.S)
    code = re.sub('""".*?"""', "", code, flags=re.S)
    lines = code.split("\n")
    # print("-------清除之后-------")
    # pprint(lines)
    for line in lines:
        line = re.sub(blank_pattern, "", line)
        line = re.sub(" +", " ", line)
        java_start = line.find("//")
        python_start = line.find("#")
        if java_start != -1:
            line = line[0:java_start]
        if python_start != -1:
            line = line[0:python_start]
        # line = re.sub(r'[\u4e00-\u9fa5].*[\u4e00-\u9fa5]', "", line, flags=re.S)
        if len(line) < limit or re.match("(import .*)|(include .*)|(from .*)", line):
            continue
        codes.append(line)
    return codes


def clean_code_line(code_line, low_limit=7, high_limit=30):
    """
    获得干净的code(不含\t，\n，连续2个以上空格，注释)
    目前注释只支持
    多行注释：/**/和三个'和三个"
    单行注释：//和#
    :param high_limit: 每行代码的最长长度，大于该长度的代码行将会被过滤
    :param low_limit: 每行代码的最短长度，小于该长度的代码行将会被过滤
    :param code_line: 源代码行
    :return: 干净的code
    """
    blank_pattern = "[\\t\\n]+"
    code_line = re.sub("/(\\*).*?(\\*)/", "", code_line, flags=re.S)
    code_line = re.sub("'''.*?'''", "", code_line, flags=re.S)
    code_line = re.sub('""".*?"""', "", code_line, flags=re.S)
    code_line = re.sub("\n+", "", code_line)
    # print("-------清除之后-------")
    # pprint(lines)
    code_line = re.sub(blank_pattern, "", code_line)
    code_line = re.sub(" +", " ", code_line)
    java_start = code_line.find("//")
    python_start = code_line.find("#")
    if java_start != -1:
        code_line = code_line[0:java_start]
    if python_start != -1:
        code_line = code_line[0:python_start]
    code_line = re.sub(r'[\u4e00-\u9fa5].*[\u4e00-\u9fa5]', "", code_line, flags=re.S)
    if len(code_line) < low_limit or len(code_line) > high_limit:
        return ""
    return code_line


def clean_with_low_frequency(documents: [], stopwords_set: set = "") -> [[]]:
    """
    将按列表存储的文档进行清洗
    :param stopwords_set: 可选参数, 如果选上，则表示自己提供停用词
    :param documents: 按列表存储的文档，列表中一个元素为一个文档
    :return: 清洗好的文档，二维列表，一行为一个文档的清洗后的词
    """
    if stopwords_set:
        my_stopwords = stopwords_set
    else:
        stopwords_file = open("../src/text/stopwords.txt")
        stopwords_string = stopwords_file.read()
        stopwords_file.close()
        my_stopwords = stopwords_string.split("\n")
    texts = list()
    for document in documents:
        text = list()
        for word in jieba.cut(document):
            word = word.lower().strip()
            if (word in my_stopwords) or re.match("\\s+", word) or re.match("\\d+", word):
                continue
            text.append(word)
        texts.append(text)
    return texts
