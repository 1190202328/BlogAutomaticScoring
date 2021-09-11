import re


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

    @staticmethod
    def text_code(filepath):
        """
        按代码为主还是文本为主来分为两份
        :param filepath: 文件路径
        :return: 两个标签列表
        """
        return

    @staticmethod
    def long_short(filepath):
        """
        按文章长短来分
        :param filepath: 文件路径
        :return: 两个标签列表
        """

    @staticmethod
    def chinese_english(filepath):
        """
        按中英文是否混杂来分
        :param filepath: 文件路径
        :return: 三个标签列表
        """

    def one_more_topics(self, indexes):
        """
        按是否为单一主题来分
        :param indexes: 序号列表
        :return: 多主题序号列表
        """
        more_topics = []
        urls = self._get_urls()
        more_topics_urls = []
        with open('../src/text/多个主题.txt', mode='r') as f:
            for line in f.readlines():
                line = re.sub('[\\t\\n]+', '', line)
                more_topics_urls.append(line)
        for index in indexes:
            if urls[index] in more_topics_urls:
                more_topics.append(index)
        return more_topics


if __name__ == '__main__':
    sd = SplitDataset('../src/text/按原创性分类.txt')
    print(len(sd.one_more_topics(list(range(651)))))
    print(sd.one_more_topics(list(range(651))))
