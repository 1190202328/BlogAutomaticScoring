import re
from datetime import date

import xlsxwriter
from bs4 import BeautifulSoup
import json
import requests
from lxml import etree
from src.SimilarityCalculator import SimilarityCalculator


class BlogAutomaticScoring:
    """
    读取学生自动打分的工具类
    """

    @staticmethod
    def get_text(url):
        """
        根据url地址获取页面内容
        :param url: 页面所在URL地址
        :return: 页面文档, 上传日期
        """
        req = requests.get(url=url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        text = ""
        pattern = "</p>|<p(.*)>|\\n"
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("div", class_="blog-content-box")
        for content in contents:
            if content.class_ == ("blog-tags-box" or "bar-content"):
                continue
            if (content.tag == "p" or "h1" or "ul") and not content.text.isspace():
                text += re.sub(pattern, "\n", content.text, count=0, flags=0) + "\n"
        text = re.sub("[ \t]", "", text)
        text = re.sub("\n+", "\n", text)
        contents = bf.find_all("span", class_="time")
        for content in contents:
            upload_date = date.fromisoformat(content.text[0:10])
            return text, upload_date

    @staticmethod
    def get_related_txt(txt_head, number):
        """
        根据标题在百度搜索相关文章，取出前number篇文章的url地址
        :param number: 需要相关文章的篇数
        :param txt_head: 文章标题
        :return: number篇文章的url的列表
        """
        req = requests.get(url="https://www.baidu.com/s?wd=" + txt_head + "&lm=1",
                           headers={'User-Agent': 'Baiduspider'})
        r = req.text
        html = etree.HTML(r, etree.HTMLParser)
        r1 = html.xpath('//h3')
        r2 = html.xpath('//*[@class="c-abstract"]')
        r3 = html.xpath('//*[@class="t"]/a/@href')
        for i in range(number):
            r11 = r1[i].xpath('string(.)')
            r22 = r2[i].xpath('string(.)')
            r33 = r3[i]
            print(r11, end='\n')
            print('------------------------')
            print(r22, end='\n')
            print(r33)
        return 1

    @staticmethod
    def get_urls(main_url):
        """
        根据学生主页面获取所有博客的url地址
        :param main_url: 主页面地址
        :return: 所有博客的ulr地址
        """
        req = requests.get(url=main_url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("a", href=True)
        urls = set()
        for content in contents:
            if re.match(".*/article/details.*", content.get("href")):
                if re.match(".*#comments|.*blogdevteam.*", content.get("href")):
                    continue
                urls.add(content.get("href"))
        return urls

    @staticmethod
    def get_main_url(url):
        """
        根据url地址返回主页的url地址
        :param url: 任意url地址
        :return: 主页的URL地址，如果找不到则返回None
        """
        req = requests.get(url=url, headers={'User-Agent': 'Baiduspider'})
        html = req.text
        bf = BeautifulSoup(html, "html.parser")
        contents = bf.find_all("a", href=True)
        for content in contents:
            if re.match("https://blog.csdn.net/\\w+", content.get("href")):
                return content.get("href")
        return None

    @staticmethod
    def calculate_score(students, limit, start_date, end_date):
        """
        自动打分,如果文章与软件构造的相似度在limit以下，则略过该文章
        :param end_date: 结束日期
        :param start_date: 开始日期
        :param limit: 相似度限制
        :param students: 学生列表
        :return: 分数字典，得分原因字典
        """
        num_topics = 3
        path = "./src/text/"
        model_related_filename = "最终"

        dictionary = SimilarityCalculator.get_dictionary(path, model_related_filename)
        corpus = SimilarityCalculator.get_corpus(path, model_related_filename)
        index = SimilarityCalculator.get_lsi_index(path, model_related_filename)
        lsi = SimilarityCalculator.get_lsi_model(corpus, dictionary, num_topics)
        student_score_dict = dict()
        student_info_dict = dict()
        print("共{0}个学生".format(len(students)))
        num = 0
        for student in students:
            num += 1
            print("正在评分第{0}个学生:\t".format(num) + student.__str__())
            scores = list()
            score = 0.0
            if student.url is None:
                student_info_dict[student] = "url地址为空\t"
                student_score_dict[student] = 0.0
                continue
            if not re.match(".*csdn.*", student.url):
                student_info_dict[student] = "不是csdn博客\t"
                student_score_dict[student] = 0.0
                continue
            urls = BlogAutomaticScoring.get_urls(BlogAutomaticScoring.get_main_url(student.url))
            for url in urls:
                document, upload_date = BlogAutomaticScoring.get_text(url)
                if upload_date < start_date or upload_date > end_date:
                    continue
                similarity = SimilarityCalculator.get_similarity(index, document, dictionary, lsi)
                if student_info_dict.get(student) is None:
                    student_info_dict[student] = url + "\t相似度:" + similarity.__str__() + "\t"
                else:
                    student_info_dict[student] = student_info_dict.get(student) + url \
                                                 + "\t相似度:" + similarity.__str__() + "\t"
                if similarity > limit:
                    scores.append(5.0)
                else:
                    scores.append(0.0)
            scores.sort(reverse=True)
            # print(scores)
            for i in range(10):
                # print(i)
                if i == len(scores):
                    break
                score += scores[i] * ((10 - i) / 10.0)
                if score >= 5:
                    score = 5.0
                    break
            student_score_dict[student] = score
        return student_score_dict, student_info_dict

    @staticmethod
    def save_scores_to_xlsx(students, student_score_dict, student_info_dict, path, xlsx_name, limit):
        """
        将成绩信息写入xlsx文档
        :param limit: 相似度限制
        :param students: 学生列表
        :param student_score_dict: 成绩词典
        :param student_info_dict: 得分信息词典
        :param path: 路径
        :param xlsx_name: 文件名称
        :return: 无
        """
        workbook = xlsxwriter.Workbook(path + xlsx_name)
        red_style = workbook.add_format({
            "font_color": "red"
        })
        worksheet = workbook.add_worksheet("sheet1")
        worksheet.write(0, 0, "学号")
        worksheet.write(0, 1, "姓名")
        worksheet.write(0, 2, "url")
        worksheet.write(0, 3, "博客成绩")
        worksheet.write(0, 4, "备注")
        i = 1
        for student in students:
            worksheet.write(i, 0, student.id)
            worksheet.write(i, 1, student.name)
            worksheet.write(i, 2, student.url)
            worksheet.write(i, 3, student_score_dict.get(student))
            student_info = student_info_dict.get(student)
            if student_info is None:
                i += 1
                continue
            student_infos = student_info.split("\t")
            j = 4
            for student_information in student_infos:
                if re.match("相似度:0\\.\\d*", student_information):
                    student_information_detail = student_information.split(":")
                    if float(student_information_detail[1]) < limit:
                        worksheet.write(i, j, student_information, red_style)
                        j += 1
                        continue
                worksheet.write(i, j, student_information)
                j += 1
            i += 1
        workbook.close()
