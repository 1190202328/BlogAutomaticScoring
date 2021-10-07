import os
import threading

lock = threading.Lock()


class SimilarityFromPMD:
    """
    通过PMD计算两行代码是否相似的类
    """

    @staticmethod
    def run_pmd():
        result = os.system("cd ../pmd/bin;./run.sh cpd --minimum-tokens 5 --files "
                           "../text/查询代码.java > ../text/返回结果.txt")
        # print(result >> 8)

    @staticmethod
    def write_codes_to_file(code1, code2):
        f = open("../pmd/text/查询代码.java", mode="w")
        f.write(code1.__str__())
        f.write("\n")
        f.write(code2.__str__())
        f.write("\n")
        f.close()

    @staticmethod
    def get_result():
        f = open("../pmd/text/返回结果.txt", mode="r")
        text = f.read()
        if text:
            return True
        else:
            return False

    @staticmethod
    def is_similar_inner(code1, code2):
        """
        判断code1和code2是否相似(不加锁)
        :param code1: 代码行1
        :param code2: 代码行2
        :return: True：相似；False：不相似
        """
        SimilarityFromPMD.write_codes_to_file(code1, code2)
        SimilarityFromPMD.run_pmd()
        return SimilarityFromPMD.get_result()

    @staticmethod
    def is_similar(code1, code2):
        """
        判断code1和code2是否相似(加锁)
        :param code1: 代码行1
        :param code2: 代码行2
        :return: True：相似；False：不相似
        """
        if lock.acquire(True):
            result = SimilarityFromPMD.is_similar_inner(code1, code2)
            lock.release()
            return result


if __name__ == '__main__':
    # code1 = "total = 0;for (int j = n;j > 0;j--){total=total+i}"
    # code2 = "a = 0; for (int k = 0;k < m;k++){a+=k}"
    # code3 = "if(a==5){a+=10;b++;c-=5;d+=1;}"
    # print(SimilarityFromPMD.is_similar(code1, code2))
    # print(SimilarityFromPMD.is_similar(code1, code3))
    # print(SimilarityFromPMD.is_similar(code2, code3))

    # code4 = '"public static void main(String[] args) {\n" +'
    # code5 = 'public static void main(String[] args) {'
    # print(SimilarityFromPMD.is_similar(code4, code5))
    SimilarityFromPMD.run_pmd()
