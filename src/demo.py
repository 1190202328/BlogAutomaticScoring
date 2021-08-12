from pprint import pprint

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

from src.model_dir.model_edu_crf import NetEDU
from src.model_dir.model_oracle_trans import NetTrans
from src.model_dir.model_rlat_uda import NetRlat

from src.test import buildPredictTree

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


def get_EDUs(text):
    """
    获得EDU列表
    :param text: 文本，要求没有换行符，是连续紧凑的文本（即：一篇文章为一行）
    :return: EDU列表
    """
    model_3 = NetEDU(768, 7, 1).cpu()
    model_1 = NetTrans(768, 2, 1).cpu()
    model_2 = NetRlat(768, 4, 4, 1).cpu()

    model_1.load_state_dict(torch.load("saved_model/pretrained_trans.pkl", map_location=torch.device('cpu')))
    model_1.eval()
    model_2.load_state_dict(torch.load("saved_model/pretrained_rlat.pkl", map_location=torch.device('cpu')))
    model_2.eval()
    model_3.load_state_dict(torch.load("saved_model/pretrained_edu.pkl", map_location=torch.device('cpu')))
    model_3.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    _, predict_leafnode_list = buildPredictTree(text, tokenizer, model_1, model_2, model_3, False)
    EDUs = list()
    for node in predict_leafnode_list:
        EDUs.append(node.sent)
    return EDUs


if __name__ == "__main__":
    # text = "据统计，这些城市去年完成国内生产总值一百九十多亿元，比开放前的一九九一年增长九成多。国务院于一九九二年先后批准了黑河、凭祥、珲春、伊宁、瑞丽等十四个边境城市为对外开放城市，同时还批准这些城市设立十四个边境经济合作区。三年多来，这些城市社会经济发展迅速，地方经济实力明显增强；经济年平均增长百分之十七，高于全国年平均增长速度，以下是来测试测试这个代码的程序。"
    # text = "由此可见，如果List的泛型重写了equals方法，则contains和indexOf方法都可以正常工作，而不需要要求参数为List中的同一个引用对象，只需要值相同即可。而将equals去掉之后，其他代码不变，发现结果如下："
    #
    # url = "https://blog.csdn.net/Louis210/article/details/117415546?spm=1001.2014.3001.5501"
    # # text, _, _ = BlogAutomaticScoring.get_text(url)
    # # text = text.replace("\n", "")
    # # print(text)
    #
    # similarity = Pretreatment.split_txt(url)
    # text = "".join(similarity["paragraphs"])
    # print(text)
    # pprint(get_EDUs(text))
    text = "99、正则表达式匹配出。www.itcast.cn。前面的<>和后面的<>是对应的，可以用此方法。100、python传参数是传值还是传址？。Python中函数参数是引用传递(注意不是值传递)。对于不可变类型(数值型、字符串、元组)，因变量不能修改，所以运算不会影响到变量自身；url='https://sycm.taobao.com/bda/tradinganaly/overview/get_summary.json?dateRange=2018-03-20%7C2018-03-20&dateType=recent1&device=1&token=ff25b109b&_=1521595613462'"
    pprint(get_EDUs(text))
