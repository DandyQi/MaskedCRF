#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 上午11:06
# @Author  : qijianwei
# @File    : data_processor.py
# @Usage: Data pre-process


import collections
import json
import random

import tensorflow as tf

import tokenization


class InputExample(object):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels


class InputFeature(object):
    def __init__(self, input_ids, input_masks, label_ids):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.label_ids = label_ids


class EnDataProcessor(object):
    def __init__(self, train_file, test_files, max_seq_length, vocab_file, do_lower_case=True):
        self.train_file = train_file
        self.test_files = None
        if len(test_files.split(",")) > 0:
            self.test_files = test_files.split(",")
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        self.label2id_map, self.id2label_map = self.get_labels()

    @staticmethod
    def get_examples(input_file, shuffle):
        examples = []
        seg_idx = 0
        with tf.gfile.GFile(input_file, "r") as f:
            line = f.readline().strip()
            while line:
                data = json.loads(line)
                if "query" in data.keys() and "tags" in data.keys():
                    query = [tokenization.convert_to_unicode(t) for t in data["query"]]
                    label = [tokenization.convert_to_unicode(l) for l in data["tags"]]
                    if len(query) == len(label):
                        examples.append(InputExample(
                            tokens=query,
                            labels=label
                        ))
                        seg_idx += 1

                line = f.readline().strip()
        f.close()
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_labels(self):
        label_set = {"[CLS]", "[SEP]"}
        all_examples = self.get_train_examples()
        for test_examples in self.get_test_examples():
            all_examples += test_examples

        for example in all_examples:
            labels = example.labels
            for label in labels:
                if label not in label_set:
                    label_set.add(label)

        label2id_map = {}
        id2label_map = {}
        labels = list(label_set)
        if "O" in labels:
            labels.remove("O")
        labels.sort()
        labels.insert(0, "O")

        for idx, label in enumerate(labels):
            label2id_map[label] = idx
            id2label_map[idx] = label

        return label2id_map, id2label_map

    def get_train_examples(self):
        examples = self.get_examples(self.train_file, True)
        return examples

    def get_test_examples(self):
        if self.test_files is None:
            return []
        test_examples = []
        for f in self.test_files:
            examples = self.get_examples(f, False)
            test_examples.append(examples)
        return test_examples

    def convert_single_example_to_feature(self, ex_idx, example):
        raw_tokens = example.tokens
        raw_labels = example.labels

        tokens = ["[CLS]"] + raw_tokens + ["[SEP]"]
        labels = ["[CLS]"] + raw_labels + ["[SEP]"]

        assert len(tokens) == len(labels)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_ids)

        label_ids = [self.label2id_map[label] for label in labels]

        input_ids_length = len(input_ids)

        if len(input_ids) < self.max_seq_length:
            input_ids += [0] * (self.max_seq_length - input_ids_length)
            input_masks += [0] * (self.max_seq_length - input_ids_length)
            label_ids += [0] * (self.max_seq_length - input_ids_length)
        else:
            input_ids = input_ids[:self.max_seq_length]
            input_masks = input_masks[:self.max_seq_length]
            label_ids = label_ids[:self.max_seq_length]

        assert len(input_ids) == self.max_seq_length
        assert len(input_ids) == len(input_masks)
        assert len(input_ids) == len(label_ids)

        if ex_idx < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("raw tokens: %s" % " ".join([tokenization.printable_text(x) for x in raw_tokens]))
            tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("labels: %s" % " ".join(labels))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        feature = InputFeature(
            input_ids=input_ids,
            input_masks=input_masks,
            label_ids=label_ids
        )
        return feature

    def file_based_convert_examples_to_features(self, examples, output_file):
        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_idx, example) in enumerate(examples):
            if ex_idx % 10000 == 0:
                tf.logging.info("Writing examples %d of %d" % (ex_idx, len(examples)))
            feature = self.convert_single_example_to_feature(
                ex_idx=ex_idx,
                example=example
            )

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_masks"] = create_int_feature(feature.input_masks)
            features["label_ids"] = create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()

    def file_based_input_fn_builder(self, input_file, is_training):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_masks": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64)
        }

        def _decode_record(record):
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["eval_batch_size"]
            d = tf.data.TFRecordDataset(input_file)
            drop_remainder = False
            if is_training:
                batch_size = params["train_batch_size"]
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                drop_remainder = True
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
            return d

        return input_fn
