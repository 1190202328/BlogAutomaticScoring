import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from anytree import Node
from collections import deque
import copy

try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

tag_to_ix_1 = {"shift": 0, "reduce": 1}
tag_to_ix_center = {'1': 0, '2': 1, '3': 2, '4': 3}
tag_to_ix_relation = {'causality': 0, 'coordination': 1, 'transition': 2, 'explanation': 3}


def createPredictLeaf(p, tokenizer, model_3):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    data = list(''.join(p))
    data.insert(0, '[CLS]')
    data.append('[SEP]')

    ans = copy.deepcopy(data)

    flag = False
    if len(data) > 512:
        flag = True

    for i in range(0, len(data), 512):
        done = False  # to make sure all tokens have been converted to ids
        # boundary checking
        if i + 512 > len(data):
            j = len(data)
        else:
            j = i + 512
        while not done:
            try:
                data[i:j] = tokenizer.convert_tokens_to_ids(data[i:j])
                # data = tokenizer.convert_tokens_to_ids(data)
                done = True
                # print(X[count])
            except KeyError as error:
                # print('x'*100)
                err = error.args[0]
                idx = data[i:j].index(err)
                # print(idx)
                data[idx] = '[UNK]'

    data_torch = torch.tensor(data, dtype=torch.long).cpu()

    logits, path = model_3(data_torch.view(1, -1))

    stack = []
    count = 1
    for j in range(1, len(path[0])):
        if (path[0][j] != 1 and path[0][j] != 2):
            max_score, idx = torch.max(logits[:, j, 1:3], -1)
            path[0][j] = idx.item() + 1
        if (j != 1) and (path[0][j] == 2 or j == len(path[0]) - 1):
            stack.append(''.join(ans[count:j]))
            count = j
        else:
            continue
    # make sure we restore the original paragraph
    if ''.join(p) != ''.join(stack):
        print(p)
        print(stack)
    # now all the EDUs have been determined
    # create node name for them
    leafnode_name_list = []
    for idx in range(len(stack)):
        leafnode_name_list.append("s" + str(idx))
    # convert EDUs to anytree type
    leafnode_list = []
    for idx in range(len(stack)):
        leafnode_list.append(
            Node(leafnode_name_list[idx], relation="", center="", leaf=True, pos="", sent=stack[idx], is_edu=True))

    leafnode_sent = []
    for i in range(len(leafnode_list)):
        leafnode_sent.append(leafnode_list[i].sent)

    return leafnode_sent, leafnode_list


def buildPredictTree(p, tokenizer, model_1, model_2, model_3, s_list=None, info=None, netG=None, trans_netG=None):
    """
        build a predict tree based on predicted EDUs
    """
    # [v1 from stack, v2 from stack, v3 from queue ] -> label(shift or reduce)
    model_1.eval()  # Transition 'shift' or 'reduce'
    model_2.eval()  # Relation
    model_3.eval()  # EDU
    # netG.eval()
    # trans_netG.eval()

    leafnode_sent, leafnode_list = createPredictLeaf(p, tokenizer, model_3)

    node_list = []
    result = []
    stack = []
    queue = deque(leafnode_sent)
    # teminate when equals sentence of stack[-1] equals to whole paragraph
    terminal = ""
    for s in leafnode_sent:
        terminal += s
    # name of internal nodes
    node_name_list = []
    for idx in range(len(leafnode_sent) - 1):
        node_name_list.append("n" + str(idx))

    # terminal condition
    count = 0
    # while stack[len(stack)-1] != terminal:
    while count < len(node_name_list):
        # number of elements in stack < 2 --> always do 'shift'
        if len(stack) < 2:
            stack.append(queue.popleft())
        # number of elements in stack >= 2
        else:
            # queue is empty --> reduce recursively till number of elements in stack == 1(root)
            if queue == deque([]):
                # predict their relation type directly cuz they will reduce to a node anyways
                # prepare 2 sentences for model input
                sent1 = tokenizer.tokenize(stack[len(stack) - 2])
                sent2 = tokenizer.tokenize(stack[len(stack) - 1])
                # insert [CLS] and [SEP] to the sentence
                sent1.insert(0, '[CLS]')
                sent1.append('[SEP]')
                sent2.insert(0, '[CLS]')
                sent2.append('[SEP]')
                # convert to bert idx
                for i in range(0, len(sent1), 512):
                    # boundary checking
                    if i + 512 > len(sent1):
                        j = len(sent1)
                    else:
                        j = i + 512
                    sent1[i:j] = tokenizer.convert_tokens_to_ids(sent1[i:j])
                for i in range(0, len(sent2), 512):
                    # boundary checking
                    if i + 512 > len(sent2):
                        j = len(sent2)
                    else:
                        j = i + 512
                    sent2[i:j] = tokenizer.convert_tokens_to_ids(sent2[i:j])
                    # sent1 = tokenizer.convert_tokens_to_ids(sent1)
                # sent2 = tokenizer.convert_tokens_to_ids(sent2)

                v1_torch = torch.tensor(sent1, dtype=torch.long).cpu()
                v2_torch = torch.tensor(sent2, dtype=torch.long).cpu()

                center, relation = model_2(v1_torch.view(1, -1), v2_torch.view(1, -1))
                # pooled = model_2(v1_torch.view(1,-1), v2_torch.view(1,-1),)
                # center, relation = netG(pooled)
                rev_tag_to_ix_center = {v: k for k, v in tag_to_ix_center.items()}
                rev_tag_to_ix_relation = {v: k for k, v in tag_to_ix_relation.items()}

                max_score, idx = torch.max(center, 1)
                predict_tag_center = rev_tag_to_ix_center[idx.item()]

                max_score, idx = torch.max(relation, 1)
                predict_tag_relation = rev_tag_to_ix_relation[idx.item()]

                if predict_tag_relation != 'coordination' and predict_tag_center == '4':
                    predict_tag_center = '3'

                label = predict_tag_relation + '_' + predict_tag_center

                node1 = node2 = None
                # if child(node1,node2) of this node is internal node
                for node in node_list:
                    if node.sent == stack[len(stack) - 2]:
                        node1 = node
                    elif node.sent == stack[len(stack) - 1]:
                        node2 = node
                # if child(node1,node2) of this node is internal node
                for node in leafnode_list:
                    if node.sent == stack[len(stack) - 2]:
                        node1 = node
                    elif node.sent == stack[len(stack) - 1]:
                        node2 = node

                if node1.leaf == True and node2.leaf == True:
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                elif node1.leaf == True:
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]
                elif node2.leaf == True:
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                else:
                    pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]

                node_list.append(Node(node_name_list[count], children=[node1, node2], relation=label.split("_")[0],
                                      center=label.split("_")[1], leaf=False, pos=pos, sent=node1.sent + node2.sent,
                                      is_edu=False))

                count += 1
                stack.pop()
                stack.pop()
                stack.append(node1.sent + node2.sent)

            else:
                # predict shift or reduce
                # prepare 3 sentences for model input
                sent1 = tokenizer.tokenize(stack[len(stack) - 2])
                sent2 = tokenizer.tokenize(stack[len(stack) - 1])
                sent3 = tokenizer.tokenize(queue[0])
                # insert [CLS] and [SEP] to the sentence
                sent1.insert(0, '[CLS]')
                sent1.append('[SEP]')
                sent2.insert(0, '[CLS]')
                sent2.append('[SEP]')
                sent3.insert(0, '[CLS]')
                sent3.append('[SEP]')
                # convert to bert idx
                # sent1 = tokenizer.convert_tokens_to_ids(sent1)
                # sent2 = tokenizer.convert_tokens_to_ids(sent2)
                # sent3 = tokenizer.convert_tokens_to_ids(sent3)

                for i in range(0, len(sent1), 512):
                    # boundary checking
                    if i + 512 > len(sent1):
                        j = len(sent1)
                    else:
                        j = i + 512
                    sent1[i:j] = tokenizer.convert_tokens_to_ids(sent1[i:j])
                for i in range(0, len(sent2), 512):
                    # boundary checking
                    if i + 512 > len(sent2):
                        j = len(sent2)
                    else:
                        j = i + 512
                    sent2[i:j] = tokenizer.convert_tokens_to_ids(sent2[i:j])
                for i in range(0, len(sent3), 512):
                    # boundary checking
                    if i + 512 > len(sent3):
                        j = len(sent3)
                    else:
                        j = i + 512
                    sent3[i:j] = tokenizer.convert_tokens_to_ids(sent3[i:j])
                v1_torch = torch.tensor(sent1, dtype=torch.long).cpu()
                v2_torch = torch.tensor(sent2, dtype=torch.long).cpu()
                v3_torch = torch.tensor(sent3, dtype=torch.long).cpu()

                score = model_1(v1_torch.view(1, -1), v2_torch.view(1, -1), v3_torch.view(1, -1))

                rev_tag_to_ix = {v: k for k, v in tag_to_ix_1.items()}

                max_score, idx = torch.max(score, 1)
                action = rev_tag_to_ix[idx.item()]

                if action == 'shift':
                    stack.append(queue.popleft())
                    continue
                elif action == 'reduce':

                    center, relation = model_2(v1_torch.view(1, -1), v2_torch.view(1, -1))
                    # center, relation = netG(pooled)
                    rev_tag_to_ix_center = {v: k for k, v in tag_to_ix_center.items()}
                    rev_tag_to_ix_relation = {v: k for k, v in tag_to_ix_relation.items()}

                    max_score, idx = torch.max(center, 1)
                    predict_tag_center = rev_tag_to_ix_center[idx.item()]

                    max_score, idx = torch.max(relation, 1)
                    predict_tag_relation = rev_tag_to_ix_relation[idx.item()]

                    if predict_tag_relation != 'coordination' and predict_tag_center == '4':
                        predict_tag_center = '3'

                    label = predict_tag_relation + '_' + predict_tag_center

                    node1 = node2 = None

                    # if child(node1,node2) of this node is internal node
                    for node in node_list:
                        if node.sent == stack[len(stack) - 2]:
                            node1 = node
                        elif node.sent == stack[len(stack) - 1]:
                            node2 = node
                    # if child(node1,node2) of this node is internal node
                    for node in leafnode_list:
                        if node.sent == stack[len(stack) - 2]:
                            node1 = node
                        elif node.sent == stack[len(stack) - 1]:
                            node2 = node

                    if node1.leaf == True and node2.leaf == True:
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                    elif node1.leaf == True:
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]
                    elif node2.leaf == True:
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[0]
                    else:
                        pos = node1.pos.split("|")[0] + "|" + node2.pos.split("|")[1]

                    node_list.append(Node(node_name_list[count], children=[node1, node2], relation=label.split("_")[0],
                                          center=label.split("_")[1], leaf=False, pos=pos, sent=node1.sent + node2.sent,
                                          is_edu=False))

                    count += 1
                    stack.pop()
                    stack.pop()
                    stack.append(node1.sent + node2.sent)

    return node_list, leafnode_list
