#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 下午2:15
# @Author  : qijianwei
# @File    : evaluate_utils.py
# @Usage: Evaluate utils


import json
import tensorflow as tf
from collections import defaultdict


def evaluate(result, id2label, eval_result_file, predict_detail_file, tokenizer, verbose=False):
    truth_tags = []
    predict_tags = []
    with tf.gfile.GFile(predict_detail_file, "w") as fout:
        for res in result:
            input_ids = res["input_ids"]
            input_masks = res["input_masks"]
            label_ids = res["label_ids"]
            predict_ids = res["predict_ids"]

            input_ids = filter_padding(input_ids, input_masks)
            label_ids = filter_padding(label_ids, input_masks)
            predict_ids = filter_padding(predict_ids, input_masks)

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            labels = [id2label[idx] for idx in label_ids]
            predicts = [id2label[idx] for idx in predict_ids]

            truth_tags.extend(labels + ["O"])
            predict_tags.extend(predicts + ["O"])

            fout.write("%s\t%s\t%s\n" % (" ".join(tokens), " ".join(labels), " ".join(predicts)))
    fout.close()

    precision, recall, f1 = get_metrics(true_seqs=truth_tags, pred_seqs=predict_tags, verbose=verbose)

    with tf.gfile.GFile(eval_result_file, "w") as fout:
        fout.write("%s" % json.dumps({
            "f1": f1,
            "precision": precision,
            "recall": recall
        }))
    fout.close()


def filter_padding(inputs, masks, valid_idx=1):
    return [i for i, m in zip(inputs, masks) if m == valid_idx]


def get_metrics(true_seqs, pred_seqs, column_format=True, verbose=True):
    if not column_format:
        true_seqs = _concat_list(true_seqs)
        pred_seqs = _concat_list(pred_seqs)
    (correct_chunks, true_chunks, pred_chunks,
     correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)

    result = get_result(correct_chunks, true_chunks, pred_chunks,
                        correct_counts, true_counts, verbose=verbose)
    return result


def _concat_list(list_of_lists, insert_separator=True):
    outputs = list()
    for lst in list_of_lists:
        outputs += lst
        if insert_separator:
            outputs += ["O"]
    return outputs


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
               correct_counts, true_counts, verbose=True):
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

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)

    res = (prec, rec, f1)
    if not verbose:
        return res

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
