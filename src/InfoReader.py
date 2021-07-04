from openpyxl import load_workbook

from src.Student import Student
from PyPDF2 import PdfFileReader, PdfFileWriter

class InfoReader:
    """
    读取信息的工具类
    """

    @staticmethod
    def get_student_info(filename):
        """
            :arg
                filename:文件名称，如"学生个人博客信息.xlsx"
            :returns
                学生的集合
        """
        workbook = load_workbook('./src/text/' + filename)
        sheets = workbook.get_sheet_names()
        booksheet = workbook.get_sheet_by_name(sheets[0])
        rows = booksheet.rows
        columns = booksheet.columns
        i = 1
        students = set()
        for row in rows:
            id = booksheet.cell(row=i, column=1).value
            name = booksheet.cell(row=i, column=2).value
            url = booksheet.cell(row=i, column=3).value
            if url is None:
                i = i + 1
                continue
            id = str(id)
            student = Student(id=id[0:10], name=name, url=url)
            students.add(student)
            i = i + 1
        return students

    @staticmethod
    def get_text_from_pdf(filename):
        # 获取一个 PdfFileReader 对象
        pdf_input = PdfFileReader(open(filename, 'rb'))
        # 获取 PDF 的页数
        page_count = pdf_input.getNumPages()
        print(page_count)
        # 返回一个 PageObject
        page = pdf_input.getPage(0)
        print(page)
        # 获取一个 PdfFileWriter 对象
        pdf_output = PdfFileWriter()
        # 将一个 PageObject 加入到 PdfFileWriter 中
        pdf_output.addPage(page)
        # 输出到文件中

