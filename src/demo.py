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
    text = "据统计，这些城市去年完成国内生产总值一百九十多亿元，比开放前的一九九一年增长九成多。国务院于一九九二年先后批准了黑河、凭祥、珲春、伊宁、瑞丽等十四个边境城市为对外开放城市，同时还批准这些城市设立十四个边境经济合作区。三年多来，这些城市社会经济发展迅速，地方经济实力明显增强；经济年平均增长百分之十七，高于全国年平均增长速度。以下是来测试测试这个代码的程序。 "
    EDUs = get_EDUs(text)
    print(EDUs)
