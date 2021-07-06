import unittest
from datetime import date

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

    def test_get_time(self):
        url = "https://blog.csdn.net/Louis210/article/details/118071148"
        upload_date = date.fromisoformat("2021-06-20")
        _, test_upload_date = BlogAutomaticScoring.get_text(url)
        self.assertEqual(upload_date, test_upload_date)


if __name__ == "__main__":
    unittest.main()
