from pprint import pprint

from src.tools import SearchWeb, GetWebResource
from src.tools.SimilarityFromBERT import SimilarityFromBERT
from src.tools.SimilarityFromPMD import SimilarityFromPMD

if __name__ == '__main__':
    low_originality_url = 'https://blog.csdn.net/shuikanshui/article/details/118295952'
    high_originality_url = 'https://blog.csdn.net/Bear922342/article/details/91049762'
    # SimilarityFromBERT.get_5d_similarities(low_originality_url)
    # SimilarityFromBERT.get_5d_similarities(high_originality_url)
    low_originality_code = 'list.add("a");'
    high_originality_code = 'students = InfoReader.get_student_info("学生个人博客信息.xlsx")'

    related_codes = SearchWeb.get_related_codes(low_originality_code, '2020-10-10')
    code_similarity = []
    for related_code in related_codes:
        if SimilarityFromPMD.is_similar(low_originality_code, related_code):
            if SimilarityFromBERT.get_similarity([low_originality_code, related_code])[:1][0][1] < 0.05:
                code_similarity.append(0)
            else:
                code_similarity.append(1)
        else:
            if SimilarityFromBERT.get_similarity([low_originality_code, related_code])[:1][0][1] > 0.95:
                code_similarity.append(1)
            else:
                code_similarity.append(0)
    pprint(related_codes)
    print(code_similarity)

    related_codes = SearchWeb.get_related_codes(high_originality_code, '2020-10-10')
    code_similarity = []
    for related_code in related_codes:
        if SimilarityFromPMD.is_similar(high_originality_code, related_code):
            if SimilarityFromBERT.get_similarity([high_originality_code, related_code])[:1][0][1] < 0.05:
                code_similarity.append(0)
            else:
                code_similarity.append(1)
        else:
            if SimilarityFromBERT.get_similarity([high_originality_code, related_code])[:1][0][1] > 0.95:
                code_similarity.append(1)
            else:
                code_similarity.append(0)
    pprint(related_codes)
    print(code_similarity)