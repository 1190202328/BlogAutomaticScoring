from datetime import datetime
from pprint import pprint

from src.tools import GetWebResource, SearchWeb

if __name__ == '__main__':
    total_urls_get = []
    with open('../../text/所有文章的url.txt', mode='r') as f:
        for line in f.readlines():
            total_urls_get.append(line[:-1])
    # print(len(total_urls_get))

    start = 49
    low_date = datetime.strptime('2022-01-01', "%Y-%m-%d")
    low_i = 0
    for i in range(start, len(total_urls_get)):
        print("{}---------url>>>".format(i) + total_urls_get[i])
        result = GetWebResource.split_txt(total_urls_get[i])
        head = result['head']
        text = result['text']
        paragraphs = result['paragraphs']
        sentences = result['sentences']
        codes = result['codes']
        update_date = result['date']
        # if datetime.strptime(update_date, "%Y-%m-%d") < low_date:
        #     print(i)
        #     print(update_date)
        #     low_date = datetime.strptime(update_date, "%Y-%m-%d")
        #     low_i = i
    # print(low_i, low_date)

        # print("---------head---------")
        # print(head)
        # # print("---------text---------")
        # # print(text)
        # # # print(similarity['text'].encode('unicode_escape').decode())
        # # print("---------paragraphs共{}个---------".format(len(paragraphs)))
        # # pprint(paragraphs)
        # # print("---------sentences共{}个---------".format(len(sentences)))
        # # pprint(sentences)
        # # print("---------codes共{}个---------".format(len(codes)))
        # # i = 1
        # # for code in codes:
        # #     print("-----------code{}-----------".format(i))
        # #     i += 1
        # #     print(code)
        print('\033[0;32;40-m <' + update_date + '> \033[0m')
        print("")
        #
        to_search = "public static void main"
        print(to_search)
        pprint(SearchWeb.get_related_codes(to_search, update_date, verbose=False, number=100))
        break
