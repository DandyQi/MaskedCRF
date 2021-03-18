#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Tianwen on 2019/3/3

This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.

IOB2:
- B = begin,
- I = inside but not the first,
- O = outside

e.g.
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O

IOBES:
- B = begin,
- E = end,
- S = singleton,
- I = inside but not the first or the last,
- O = outside

e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O

prefix: IOBES
chunk_type: PER, LOC, etc.
"""

import json
import os
from collections import defaultdict, Counter


def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag in {'O', ""}:
        return 'O', None
    tags = chunk_tag.split('-')
    if len(tags) == 2:
        return tags
    else:
        return 'O', None


def is_previous_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def is_current_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    判断当前tag是否为一个starting tag，例如：
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> False

    (B-PER, B-PER) -> True
    (B-LOC, B-PER)  -> True
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    tp = float(tp)
    p = float(p)
    t = float(t)
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_previous_chunk_end(prev_true_tag, true_tag)
            pred_end = is_previous_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_current_chunk_start(prev_true_tag, true_tag)
        pred_start = is_current_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)


def get_result(correct_chunks, true_chunks, pred_chunks,
               correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)

    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type

    # print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    # print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')

    # print("accuracy: %6.2f%%; (non-O)" % (100 * nonO_correct_counts / nonO_true_counts))
    if sum_true_counts == 0:
        print("[Tagging] %6.2f; " % 0.0)
    else:
        print("[Tagging] %6.2f; " % (100 * sum_correct_counts / sum_true_counts))
    print("precision: %6.2f; recall: %6.2f; FB1: %6.2f" % (prec, rec, f1))

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        print("Task: %s\tPrecision: %6.2f; recall: %6.2f; FB1: %6.2f\t(support:  %d)"
              % (t, prec, rec, f1, pred_chunks[t]))

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this


def evaluate(true_seqs, pred_seqs, column_format=True, verbose=True):
    if not column_format:
        true_seqs = _concat_list(true_seqs)
        pred_seqs = _concat_list(pred_seqs)
    (correct_chunks, true_chunks, pred_chunks,
     correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)

    result = get_result(correct_chunks, true_chunks, pred_chunks,
                        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result


def _concat_list(list_of_lists, insert_separator=True):
    outputs = list()
    for lst in list_of_lists:
        outputs += lst
        if insert_separator:
            outputs += ["O"]
    return outputs


def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []

    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)


def _convert(inputs, tag_map, by_row, correct_singletons, separator="O"):
    outputs = list()
    for tag_encodings in inputs:
        original_tags = list(map(lambda x: tag_map[x], tag_encodings))
        if correct_singletons:
            original_tags = remove_singleton(original_tags)
        if by_row:
            outputs.append(original_tags)
        else:
            outputs += original_tags
            outputs.append(separator)
    return outputs


def _filter(inputs, valid_tags):
    outputs = [tags for (tags, is_valid) in zip(inputs, valid_tags) if is_valid]
    print("Inputs: {}\t Outputs after filtering: {}".format(len(inputs), len(outputs)))
    return outputs


def _unpad_seq(list_seq, lengths, max_length):
    """把一个pad之后的序列反pad，使之回到pad之前的状态。
    如果序列的原始长度大于max_length，则其被截断的部分无法恢复。

    Args:
        list_seq: list, pad之后的序列
        lengths: list, 序列的原始长度
        max_length: int, pad序列的最大长度

    Returns:
        反pad之后的序列
    """
    ret = list()
    for i, seq in enumerate(list_seq):
        actual_length = min(lengths[i], max_length)
        # print("Input length={}\t actual length={}".format(len(seq), actual_length))
        ret.append(seq[:actual_length])
    return ret


def convert_tags(labels, preds, tag_map, lengths=None, by_row=True, separator="O", correct_singletons=False,
                 tag_mask=None, do_unpad=True, max_length=30):
    """ 把tags的编码转换为对应的原始tag名
    支持以行格式输出（by_row=True）和以列格式输出（by_row=False）。
    在进行编码转换之前，还可以进行如下操作（可选）：

    * 过滤样本（tag_mask不为None）
    * 修正单字槽位（correct_singletons为True）
    * 去除padding（do_unpad为True）

    --------------------------------------------------------
    # by_row=True, 每条样本的tags之间以换行\n区分

    O B-to_city I-to_city O O O
    O O O O O O
    O O O B-from_city I-from_city O B-to_city I-to_city O O
    ...
    --------------------------------------------------------
    # by_row=False, 每条样本的tags之间以空行separator区分

    O
    B-to_city
    I-to_city
    O
    O
    <separator>
    O
    O
    ...
    --------------------------------------------------------

    Args:
        labels: list (list of list of str), 标注的tag编码
        preds: list (list of list of str), 预测的tag编码
        tag_map: dict, 从tag编码到原始tag名的映射表
        lengths: list (list of int), 原始query长度
        by_row: bool, 如果为True，表示按行输出，格式为list of list of str;
            如果为False，则表示按列输出，格式为list of str
        separator: str, 当以列格式输出时，两条不同query之间的分隔符，默认为"O"
        correct_singletons: bool, 是否自动修正singleton tag（单字槽位）
        tag_mask: list of bool of bool convertibles。如果valid_tags不为空，
            则表示需要过滤掉无效的tags
        do_unpad: bool, 是否对输入的labels和preds进行unpadding
        max_length: int, query的最大长度

    Returns:
        编码后的tags列表
    """

    if tag_mask:
        if len(tag_mask) != len(labels):
            print("Warning: len(tag_mask)={}, which does not match len(labels)={}".format(len(tag_mask), len(labels)))
        else:
            labels = _filter(labels, tag_mask)
            preds = _filter(preds, tag_mask)
            lengths = _filter(lengths, tag_mask)

    if do_unpad:
        if len(lengths) != len(labels):
            print("Warning: len(lengths)={}, which does not match len(labels)={}".format(len(lengths), len(labels)))
        else:
            labels = _unpad_seq(labels, lengths, max_length)
            preds = _unpad_seq(preds, lengths, max_length)

    output_labels = _convert(labels, tag_map, by_row, False, separator)
    output_preds = _convert(preds, tag_map, by_row, correct_singletons, separator)
    return output_labels, output_preds


def save_slot_badcases(queries, labels, preds, save_file, print_diff=True):
    """把tagging的badcase保存至文件。

    * 如果参数print_diff为True，则储存格式为：
    ---------------------------
    Query   Label   Prediction
    ---------------------------
    帮	    O	        -
    我	    O	        -
    搜	    O	        -
    索	    O	        -
    张	    B-from_city	O
    掖	    I-from_city	O
    到	    O	        -
    嘉	    B-to_city	-
    峪	    I-to_city	-
    关	    I-to_city	-
    的	    O	        -
    车	    O	        -
    ---------------------------

    * 如果参数print_diff为False，则储存格式为：
    ---------------------------
    Query   Label   Prediction
    ---------------------------
    帮	    O	        O
    我	    O	        O
    搜	    O	        O
    索	    O	        O
    张	    B-from_city	O
    掖	    I-from_city	O
    到	    O	        O
    嘉	    B-to_city	B-to_city
    峪	    I-to_city	I-to_city
    关	    I-to_city	I-to_city
    的	    O	        O
    车	    O	        O
    ---------------------------

    Args:
        queries: list, 原始query
        labels: list, tag标签
        preds: list, tag预测
        save_file: str, 保存文件路径
        print_diff: bool, 是否突出显示标注与预测不同那些的tags
    """
    with open(save_file, 'w', encoding="UTF-8") as fobj:
        for j in range(len(queries)):
            query = queries[j]
            label = labels[j]
            pred = preds[j]
            if label == pred:
                continue
            if not (min(len(query), 30) == len(label) == len(pred)):
                print("Warning: lengths do not match: \n{}\t{}\t{}".format(query, label, pred))
                continue

            fobj.write(query + "\n")
            for i in range(len(label)):
                if print_diff:
                    if label[i] == pred[i]:
                        fobj.write("\t".join([query[i], label[i], "-"]) + "\n")
                    else:
                        fobj.write("\t".join([query[i], label[i], pred[i]]) + "\n")
                else:
                    fobj.write("\t".join([query[i], label[i], pred[i]]) + "\n")
            fobj.write("\n\n")


def debug_tagging_predictions(queries, labels, preds, save_file, print_diff=True):
    """把tagging的badcase保存至文件。

    * 如果参数print_diff为True，则储存格式为：
    ---------------------------
    Query   Label   Prediction
    ---------------------------
    帮	    O	        -
    我	    O	        -
    搜	    O	        -
    索	    O	        -
    张	    B-from_city	O
    掖	    I-from_city	O
    到	    O	        -
    嘉	    B-to_city	-
    峪	    I-to_city	-
    关	    I-to_city	-
    的	    O	        -
    车	    O	        -
    ---------------------------

    * 如果参数print_diff为False，则储存格式为：
    ---------------------------
    Query   Label   Prediction
    ---------------------------
    帮	    O	        O
    我	    O	        O
    搜	    O	        O
    索	    O	        O
    张	    B-from_city	O
    掖	    I-from_city	O
    到	    O	        O
    嘉	    B-to_city	B-to_city
    峪	    I-to_city	I-to_city
    关	    I-to_city	I-to_city
    的	    O	        O
    车	    O	        O
    ---------------------------

    Args:
        queries: list, 原始query
        labels: list, tag标签
        preds: list, tag预测
        save_file: str, 保存文件路径
        print_diff: bool, 是否突出显示标注与预测不同那些的tags
    """
    with open(save_file, 'w', encoding="UTF-8") as fobj:
        for j in range(len(queries)):
            query = queries[j]
            label = labels[j]
            pred = preds[j]

            if not (min(len(query), 30) == len(label) == len(pred)):
                print("{}\t{}\t{}".format(query, label, pred))
                continue

            fobj.write(query + "\n")
            for i in range(len(label)):
                if print_diff:
                    if label[i] == pred[i]:
                        fobj.write("\t".join([query[i], label[i], "-"]) + "\n")
                    else:
                        fobj.write("\t".join([query[i], label[i], pred[i]]) + "\n")
                else:
                    fobj.write("\t".join([query[i], label[i], pred[i]]) + "\n")
            fobj.write("\n\n")


def remove_singleton(tags):
    """
    修正singleton tag（即单字槽位）为"O"并返回。
    例如：

    ["B-from_city", "I-from_city", "O", "B-to_city", "O", "O", "O"]
        =>
    ["B-from_city", "I-from_city", "O", "O", "O", "O", "O"]

    上面例子中"B-to_city"前后都是"O"，此时我们称其为一个singleton tag。

    Args:
        tags: list, 一条query对应的tags

    Returns:
        list, 修正了singleton之后的tags
    """
    labels = list()
    for tag in tags:
        if tag == "O":
            labels.append(tag)
        else:
            labels.append(tag.split("-")[1])

    for i, lab in enumerate(labels):
        if i == 0:
            if lab != "O" and labels[1] == "O":
                tags[0] = "O"
        elif i == len(labels) - 1:
            if lab != "O" and labels[i - 1] == "O":
                tags[-1] = "O"
        else:
            if lab != "O" and labels[i - 1] == "O" and labels[i + 1] == "O":
                tags[i] = "O"
    return tags


def test_remove_singleton():
    tags1 = ["B-time", "B-from_city", "O", "B-to_city", "O", "O", "O"]
    tags2 = ["B-from_city", "O", "O", "O", "O", "O", "B-time", "B-from_city"]
    tags3 = ["O", "O", "B-time", "B-from_city", "O", "O", "O", "B-to_city"]
    print(remove_singleton(tags1))
    print(remove_singleton(tags2))
    print(remove_singleton(tags3))


def check_ignored_slots(tags, slots_to_ignore):
    if slots_to_ignore is None:
        return tags
    ret = list()
    for tag in tags:
        flag_ignore = False
        for slot in slots_to_ignore:
            if tag == "B-" + slot or tag == "I-" + slot:
                flag_ignore = True
                break
        if flag_ignore:
            ret.append("O")
        else:
            ret.append(tag)
    return ret


def test_check_ignored_slots():
    tags = ["O", "B-time", "I-time", "I-time", "I-time", "I-time", "O", "B-to_city", "I-to_city", "O"]
    slots_to_ignore = ("time",)
    ret = check_ignored_slots(tags, slots_to_ignore)
    for x, y in zip(tags, ret):
        print("{}\t{}".format(x, y))


def eval_houyi(query_label_file, query_pred_file, badcase_file=None, slots_to_ignore=None, print_diff=True):
    """
    计算后羿系统的tagging准确率

    Args:
        query_label_file: str, 标注文件，两列，分别是query和tags
        query_pred_file: str, 预测文件，两列，分别是query和preds
        badcase_file: str, 如果不为None，将badcase保存至此文件
        print_diff: bool, 是否突出显示标注与预测不同那些的tags
    """
    queries = list()
    labels_by_row = list()
    preds_by_row = list()

    labels_by_col = list()
    preds_by_col = list()

    with open(query_label_file, 'r', encoding="UTF-8") as f1, open(query_pred_file, 'r', encoding="UTF-8") as f2:
        for line in f1:
            query, label = line.strip().split("\t")
            queries.append(query)
            labels_by_row.append(check_ignored_slots(label.split(), slots_to_ignore))

        for line in f2:
            query, pred = line.strip().split("\t")
            preds_by_row.append(check_ignored_slots(pred.split(), slots_to_ignore))

    for i in range(len(queries)):
        if not len(queries[i]) == len(labels_by_row[i]) == len(preds_by_row[i]):
            print(queries[i], labels_by_row[i], preds_by_row[i])
            print(len(queries[i]), len(labels_by_row[i]), len(preds_by_row[i]))
        else:
            labels_by_col += labels_by_row[i]
            labels_by_col += ["O"]

            preds_by_col += preds_by_row[i]
            preds_by_col += ["O"]
    print(len(labels_by_col))
    print(len(preds_by_col))
    evaluate(labels_by_col, preds_by_col)

    if badcase_file:
        save_slot_badcases(queries, labels_by_row, preds_by_row, badcase_file, print_diff=print_diff)


def get_slot_tokens(query, tags, slots):
    start_index = list()
    end_index = list()
    outputs = dict()
    for slot in slots:
        outputs.update({slot: list()})
    for i in range(len(tags)):
        prefix, label = split_tag(tags[i])

        if i == 0:
            if prefix != "O":
                start_index.append(i)

        else:
            if is_current_chunk_start(tags[i - 1], tags[i]):
                start_index.append(i)
            if is_previous_chunk_end(tags[i - 1], tags[i]):
                end_index.append(i - 1)
            if i == len(tags) - 1 and prefix != "O":
                end_index.append(i)

    assert (len(start_index) == len(end_index))
    for start, end in zip(start_index, end_index):
        token = query[start:end + 1]
        labels = tags[start: end + 1]
        labels = list(set(map(lambda x: x.split("-")[1], labels)))
        assert (len(labels) == 1)
        outputs[labels[0]].append(token)
    return outputs


def test_get_slot_tokens():
    slots = ("from_city", "to_city", "time")

    query1 = "明天到北京的飞机票"
    tags1 = ["B-time", "I-time", "O", "B-to_city", "I-to_city", "O", "O", "O", "O"]
    d1 = {'from_city': [], "time": ["明天"], "to_city": ["北京"]}
    output1 = get_slot_tokens(query1, tags1, slots)
    print(output1)

    query2 = "帮我查一下淄博到威海"
    tags2 = ["O", "O", "O", "O", "O", "B-from_city", "I-from_city", "O", "B-to_city", "I-to_city"]
    d2 = {'time': [], "from_city": ["淄博"], "to_city": ["威海"]}
    output2 = get_slot_tokens(query2, tags2, slots)
    print(output2)

    query3 = "帮我查一下淄博到威海到南京"
    tags3 = ["O", "O", "O", "O", "O", "B-from_city", "I-from_city", "O", "B-to_city", "I-to_city", "O", "B-to_city",
             "I-to_city"]
    d3 = {'time': [], "from_city": ["淄博"], "to_city": ["威海", "南京"]}
    output3 = get_slot_tokens(query3, tags3, slots)
    print(output3)

    assert (output1 == d1)
    assert (output2 == d2)
    assert (output3 == d3)
    print("get_slot_tokens() test passed!")


def save_slot_stats(inputs, slots=("from_city", "to_city", "time")):
    slot_tokens = dict()
    for slot in slots:
        slot_tokens.update({slot: list()})

    with open(inputs, 'r', encoding="UTF-8") as fin:
        for line in fin:
            sample = json.loads(line.strip())
            if "tags" not in sample:
                continue
            query = sample["query"]
            tags = sample["tags"]
            token_dict = get_slot_tokens(query, tags, slots)

            for slot in slots:
                slot_tokens[slot] += token_dict[slot]

    output_dir = os.path.split(inputs)[0]
    for slot in slots:
        filename = os.path.join(output_dir, "count_{}.txt".format(slot))
        slot_list = slot_tokens[slot]
        count = sorted(Counter(slot_list).items(), key=lambda x: x[1], reverse=True)
        with open(filename, 'w', encoding="UTF-8") as f:
            for word, count in count:
                f.write("{}\t{}\n".format(word, count))


def result_parse(result_file):
    with open(result_file, "r") as res_file, \
            open("data/ner/query_label.txt", "w", encoding="utf-8") as label_file, \
            open("data/ner/query_pred.txt", "w", encoding="utf-8") as pred_file:
        line = res_file.readline()
        while line:
            query, label, pred = line.strip().split("\t")

            if "回家" in query:
                line = res_file.readline()
                continue

            if len(query) > 19:
                query = query[:19]

            if len(label.split()) > 19:
                label = " ".join(label.split()[:19])

            pred = " ".join(pred.split()[1: len(query) + 1])

            label_file.write("%s\t%s\n" % (query, label))
            pred_file.write("%s\t%s\n" % (query, pred))

            line = res_file.readline()


def diff_eval(result_file_1, result_file_2, diff_file):
    clean_query = set()
    with open("data/val/qq/clean_gsb_qq.txt", "r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if line not in clean_query:
                clean_query.add(line)
    print("clean query size: %d" % len(clean_query))
    fin.close()

    eval_count = 0
    diff_count = 0
    with open(result_file_1, "r") as fin1, open(result_file_2, "r") as fin2, open(diff_file, "w") as fout:
        for line_1, line_2 in zip(fin1.readlines(), fin2.readlines()):
            line_1 = line_1.strip()
            line_2 = line_2.strip()

            query_1, tags_1 = line_1.split("\t")
            query_2, tags_2 = line_2.split("\t")

            tags_1 = tags_1.split(" ")
            tags_2 = tags_2.split(" ")

            if query_1 != query_2:
                continue

            if len(query_1) > 32:
                continue

            if query_1 not in clean_query:
                continue

            eval_count += 1
            if tags_1 != tags_2:
                s = "%s\n" % query_1
                # for token, tag_1, tag_2 in zip(query_1, tags_1, tags_2):
                #     s += "%s\t%s\t%s\n" % (token, tag_1, tag_2)
                # s += "\n============\n\n"
                fout.write(s)
                diff_count += 1

    fin1.close(), fin2.close(), fout.close()

    print("Evaluate %d query, %d are different" % (eval_count, diff_count))


def clean_result(input_file, output_file):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin.readlines():
            line = line.strip()

            query, tags = line.split("\t")
            tags = tags.split(" ")
            new_tags = []
            for token, tag in zip(query, tags):
                if (tag == "B-user_name" or tag == "I-user_name") and token == "打":
                    tag = "O"
                new_tags.append(tag)
            fout.write("%s\t%s\n" % (query, " ".join(new_tags)))

    fin.close()
    fout.close()


def eval_classification(input_file, badcase_file):
    count = 0
    correct = 0.0
    with open(input_file, "r") as fin, open(badcase_file, "w") as fout:
        for line in fin.readlines():
            line = line.strip()
            q, t, p = line.split("\t")
            if t == p:
                correct += 1
            else:
                fout.write(line + "\n")
            count += 1

    print("accuracy: %.4f" % (correct / count))
    fin.close()
    fout.close()


if __name__ == '__main__':
    # pass
    # usage:     conlleval < file
    # evaluate_conll_file(sys.stdin)

    # pass
    # test_get_slot_tokens()
    # save_slot_stats("transportTicket/trainset/data-v0.29.json")

    # result_parse("model/qq/predict_results_1.txt")
    # eval_houyi("data/ner/query_label.txt", "data/ner/query_pred.txt", "data/ner/badcase.txt", ("time",))
    # eval_houyi("data/ner/query_label.txt", "data/ner/query_pred.txt", "data/ner/badcase.txt")
    # test_check_ignored_slots()

    # clean_result("data/val/qq/bert_result_corrected2.txt", "data/val/qq/bert_result_corrected3.txt")
    # diff_eval("data/val/qq/bert_result_corrected3.txt", "data/val/qq/lstm_result.txt", "data/val/qq/diff_result.txt")
    # eval_classification("data/val/ticketClassification/predict_results.txt",
    #                     "data/val/ticketClassification/badcase.txt")
    # map_eval("data/val/map/test6/map_bilstm_15.txt", "data/val/map/test6/badcases.txt")

    # eval_truncation("model/multitask/nlp_truncated/predict_results_1.txt")
    pass
