import unittest

from src.BlogAutomaticScoring import BlogAutomaticScoring
from src.SimilarityCalculator import SimilarityCalculator


class testSimilarityCalculator(unittest.TestCase):
    def test_get_similarity(self):
        path = "../src/text/"
        limit = 0.9
        document1, _ = BlogAutomaticScoring.get_text(url="https://blog.csdn.net/Louis210/article/details/117415546")
        document2, _ = BlogAutomaticScoring.get_text(url="https://blog.csdn.net/Louis210/article/details/118071148")
        document3, _ = BlogAutomaticScoring.get_text(url="https://blog.csdn.net/zhuzyibooooo/article/details/118230353")
        document4, _ = BlogAutomaticScoring.get_text(url="https://blog.csdn.net/zP1nG/article/details/118418217")
        document5 = "关键词：OOP、ADT、API、Java、static、final、Set、Map、ArrayList、软件构造Software " \
                    "Construction、父类与子类、继承与复用、抽象类与实体类、陈惠鹏、规约、委派、面向对象、软件测试、二手烟、正确性、健壮性、可维护性、可复用性、可扩展性。 "
        document6 = "在活动中心调bug到十点没调完，被赶出来了接着调。我爱软件构造，软件构造真尼玛有意思。"
        dictionary = SimilarityCalculator.get_dictionary(path, "测试词典")
        corpus = SimilarityCalculator.get_corpus(path, "测试词袋向量")
        index = SimilarityCalculator.get_lsi_index(path, "测试索引")
        lsi = SimilarityCalculator.get_lsi_model(corpus, dictionary, 2)
        self.assertEqual(True, SimilarityCalculator.get_similarity(index, document1, dictionary, lsi) > limit)
        self.assertEqual(False, SimilarityCalculator.get_similarity(index, document2, dictionary, lsi) > limit)
        self.assertEqual(False, SimilarityCalculator.get_similarity(index, document3, dictionary, lsi) > limit)
        self.assertEqual(False, SimilarityCalculator.get_similarity(index, document4, dictionary, lsi) > limit)
        self.assertEqual(True, SimilarityCalculator.get_similarity(index, document5, dictionary, lsi) > limit)
        self.assertEqual(False, SimilarityCalculator.get_similarity(index, document6, dictionary, lsi) > limit)