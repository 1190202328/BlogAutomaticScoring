import re

import jieba

from src.Pretreatment import Pretreatment


class SplitDataset:
    """
    分解数据集的工具类
    """

    def __init__(self, filepath):
        """
        :param filepath: 文件路径
        """
        self.filepath = filepath

    def _get_urls(self):
        """
        获取url列表
        :return:
        """
        urls = []
        with open(self.filepath, mode='r') as f:
            for line in f.readlines():
                line = re.sub('\\t+', '\\t', line)
                line = line.split('\t')
                if len(line) <= 2:
                    break
                urls.append(line[1].replace('\n', ''))
        return urls

    def data_split(self, indexes, language_rate, length, code_rate):
        """
        按中文为主，英文为主，中英文混杂来分
        :param code_rate: 代码占比高与code_rate则为代码为主，否则为文本为主
        :param length: 文章词语数大于length则为长文章，否则为短文章
        :param language_rate: 中文占比language_rate以上则认为是全中文；英文占比language_rate以上则认为是全英文；其余为混杂。
        :param indexes: 序号列表
        :return: 词典，其中chinese：中文为主；english：英文为主；mixed：中英文混合；long：长文章；short：短文章；
        text：文本为主；code：代码为主
        """
        data_set = dict()
        chinese = []
        english = []
        mixed = []
        long = []
        short = []
        text_main = []
        code_main = []
        urls = self._get_urls()
        chinese_pattern = r'[\u4e00-\u9fa5].*'
        for index in indexes:
            print('<<<{}>>>'.format(index))
            chinese_count = 0
            total_count = 0
            result = Pretreatment.split_txt(urls[index], verbose=False)
            if result is None:
                print(urls[index])
                return None
            text = result.get('text')
            text = text.replace('\n', '')
            for word in jieba.cut(text):
                if not re.match(r'[\u4e00-\u9fa5A-Za-z].*', word):
                    continue
                total_count += 1
                if re.match(chinese_pattern, word):
                    chinese_count += 1
            if chinese_count / total_count > language_rate:
                chinese.append(index)
            elif 1 - chinese_count / total_count > language_rate:
                english.append(index)
            else:
                mixed.append(index)
            if total_count > length:
                long.append(index)
            else:
                short.append(index)
            sentences = result.get('sentences')
            codes = result.get('codes')
            if not codes:
                text_main.append(index)
                continue
            code_count = 0
            for code in codes:
                for line in code.split('\n'):
                    if re.match('\\s+', line):
                        continue
                    code_count += 1
            if code_count / (len(sentences) + code_count) > code_rate:
                code_main.append(index)
            else:
                text_main.append(index)
        data_set['chinese'] = chinese
        data_set['english'] = english
        data_set['mixed'] = mixed
        data_set['long'] = long
        data_set['short'] = short
        data_set['text'] = text_main
        data_set['code'] = code_main
        return data_set

    def one_more_topics(self, indexes: list) -> tuple[list, list]:
        """
        按是否为单一主题来分
        :param indexes: 序号列表
        :return: multi_topics, single_topic；共两个列表
        """
        multi_topics = []
        single_topic = []
        urls = self._get_urls()
        more_topics_urls = []
        with open('../src/text/多个主题.txt', mode='r') as f:
            for line in f.readlines():
                line = re.sub('[\\t\\n]+', '', line)
                more_topics_urls.append(line)
        for index in indexes:
            if urls[index] in more_topics_urls:
                multi_topics.append(index)
            else:
                single_topic.append(index)
        return multi_topics, single_topic


if __name__ == '__main__':
    total_list = list(range(651))
    sd = SplitDataset('../src/text/按原创性分类.txt')
    language_rate = 0.9
    length = 300
    code_rate = 0.3
    # print(len(sd.one_more_topics(total_list)))
    # print(sd.one_more_topics(total_list))
    data_set = sd.data_split(total_list, language_rate, length, code_rate)
    print(len(data_set['chinese']))
    print(len(data_set['english']))
    print(len(data_set['mixed']))
    print(len(data_set['long']))
    print(len(data_set['short']))
    print(len(data_set['text']))
    print(len(data_set['code']))
