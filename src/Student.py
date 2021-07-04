class Student:
    """
    学生类 immutable
    """

    def __init__(self, id, name, url):
        self.__id = id
        self.__name = name
        self.__url = url

    @property
    def id(self):
        return self.__id

    @property
    def name(self):
        return self.__name

    @property
    def url(self):
        return self.__url

    @property
    def score(self):
        return self.__score

    def __str__(self):
        return "id={0}\tname={1}\turl={2}".format(self.__id, self.__name, self.__url)
