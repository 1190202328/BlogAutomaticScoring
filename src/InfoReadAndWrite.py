from pprint import pprint


class InfoReadAndWrite:
    """
    从爬虫上读取的信息读写工具类
    """

    @staticmethod
    def get_urls(path, filename):
        urls = list()
        f = open(path + filename, "r")
        for line in f.readlines():
            urls.append(line[:-1])
        f.close()
        return urls

    @staticmethod
    def get_result(url):
        return 0


if __name__ == '__main__':
    path = "../src/text/"
    filename = "url总集.txt"
    urls = InfoReadAndWrite.get_urls(path, filename)
    print(len(urls))
    pprint(urls)
