import unittest
from src.BlogAutomaticScoring import BlogAutomaticScoring


class testBlogAutomaticScoring(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_urls(self):
        main_url = "https://blog.csdn.net/Louis210"
        urls = set()
        urls.add("https://blog.csdn.net/Louis210/article/details/118071148")
        urls.add("https://blog.csdn.net/Louis210/article/details/117415546")
        urls.add("https://blog.csdn.net/Louis210/article/details/116278734")
        self.assertEqual(urls, BlogAutomaticScoring.get_urls(main_url=main_url))

    def test_get_main_url(self):
        url = "https://blog.csdn.net/Louis210/article/details/118071148"
        main_url = "https://blog.csdn.net/Louis210"
        self.assertEqual(main_url, BlogAutomaticScoring.get_main_url(url=url))
        self.assertEqual(main_url, BlogAutomaticScoring.get_main_url(url=main_url))


if __name__ == "__main__":
    unittest.main()
