#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 上午11:01
# @Author  : qijianwei
# @File    : train_and_eval.py
# @Usage: Bert + CRF / MCRF train and evaluate


import glob
import json
import logging
import os
import shutil
import time

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig

import modeling
import optimization
from data_processor import DataProcessor
from evaluate_utils import evaluate
from masked_crf import MaskedCRF

logger = logging.getLogger()
while logger.handlers:
    logger.handlers.pop()

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None, "Bert config file")

flags.DEFINE_string("train_file", None, "Train data file")

flags.DEFINE_string("dev_file", None, "Dev data file")

flags.DEFINE_string("test_file", None, "Test data file")

flags.DEFINE_string("data_format", None, "Data format, support rows or cols")

flags.DEFINE_integer("max_seq_length", 128, "Max sequence length")

flags.DEFINE_string("vocab_file", None, "Tokenizer vocab file")

flags.DEFINE_bool("do_lower_case", False, "Whether to use lowercase")

flags.DEFINE_string("output_dir", None, "Model output dir")

flags.DEFINE_integer("train_batch_size", 32, "Train batch size")

flags.DEFINE_integer("eval_batch_size", 8, "Evaluate batch size")

flags.DEFINE_integer("epoch_num", 1, "Train epoch number")

flags.DEFINE_string("init_checkpoint", None, "Initialize checkpoint")

flags.DEFINE_integer("min_train_steps", 10000, "Map train steps")

flags.DEFINE_float("learning_rate", 2e-5, "Learning rate")

flags.DEFINE_float("lr_decay", 1.0, "Learning rate decay factor")

flags.DEFINE_bool("train_mask", False, "Whether to mask when train")

flags.DEFINE_bool("eval_mask", False, "Whether to mask when evaluate")

flags.DEFINE_string("gpu_idx", "0", "GPU idx")


class BestCheckpointsExporter(tf.estimator.BestExporter):

    def __init__(self,
                 name='best_exporter',
                 serving_input_receiver_fn=None,
                 event_file_pattern='eval/*.tfevents.*',
                 compare_fn=None,
                 assets_extra=None,
                 as_text=False,
                 exports_to_keep=5,
                 best_checkpoint_path=None):
        tf.estimator.BestExporter.__init__(self, name, serving_input_receiver_fn, event_file_pattern, compare_fn,
                                           assets_extra, as_text,
                                           exports_to_keep)
        self.best_checkpoint_path = best_checkpoint_path

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        if self._best_eval_result is None or self._compare_fn(self._best_eval_result, eval_result):
            tf.logging.info('Exporting a better model ({} instead of {})...'
                            .format(eval_result, self._best_eval_result))
            # copy the checkpoints files *.meta *.index, *.data* each time there is a better result
            # no cleanup for max amount of files here

            best_ckpt_path = os.path.join(self.best_checkpoint_path, str(int(time.time())))
            tf.gfile.MakeDirs(best_ckpt_path)

            for name in glob.glob(checkpoint_path + '.*'):
                shutil.copy(name, os.path.join(best_ckpt_path, os.path.basename(name)))

            # also save the text file used by the estimator api to find the best checkpoint
            with open(os.path.join(self.best_checkpoint_path, "best_checkpoint.txt"), 'w') as f:
                content = {
                    "best_checkpoint_path": os.path.join(best_ckpt_path, os.path.basename(checkpoint_path))
                }
                f.write("%s" % json.dumps(content, ensure_ascii=False))
            self._best_eval_result = eval_result

            self._garbage_collect_exports(self.best_checkpoint_path)
        else:
            tf.logging.info('Keeping the current best model ({} instead of {}).'
                            .format(self._best_eval_result, eval_result))


def model_fn_builder(model_config, init_checkpoint, learning_rate, lr_decay, num_output,
                     max_seq_length, mask_crf):
    def model_fn(features, labels, mode, params):
        if labels is None and params is None:
            pass
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_masks = features["input_masks"]
        label_ids = features["label_ids"]
        lengths = tf.reduce_sum(input_masks, axis=-1)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_masks
        )

        # [batch size, sequence length, hidden size]
        input_tensor = model.get_sequence_output()

        input_tensor = tf.layers.dense(input_tensor, num_output, activation=tf.nn.relu)
        logits = tf.reshape(input_tensor, shape=[-1, max_seq_length, num_output])

        loss, per_example_loss, predict_ids = mask_crf.decode(
            logits=logits,
            label_ids=label_ids,
            lengths=lengths
        )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.debug("**** All Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.debug("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss=loss,
                init_lr=learning_rate,
                lr_decay_factor=lr_decay
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "eval_loss": tf.metrics.mean(per_example_loss)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "input_ids": input_ids,
                    "input_masks": input_masks,
                    "label_ids": label_ids,
                    "predict_ids": predict_ids
                }
            )

    return model_fn


def serving_fn():
    input_ids = tf.placeholder(tf.int64, [None, None], name="input_ids")
    input_mask = tf.placeholder(tf.int64, [None, None], name="input_mask")
    label_ids = tf.placeholder(tf.int64, [None, None], name="label_ids")

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "input_ids": input_ids,
        "input_mask": input_mask,
        "label_ids": label_ids,

    })()
    return input_fn


def _loss_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.

  Both evaluation results should have the values for MetricKeys.LOSS, which are
  used for comparison.

  Args:
    best_eval_result: best eval metrics.
    current_eval_result: current eval metrics.

  Returns:
    True if the loss of current_eval_result is smaller; otherwise, False.

  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
    default_key = "loss"
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    return best_eval_result[default_key] > current_eval_result[default_key]


def construct_estimator(output_dir, save_checkpoint_steps, model_config, init_checkpoint, learning_rate,
                        max_seq_length, use_mask, label2idx_map, num_output,
                        train_batch_size, eval_batch_size, lr_decay):
    mask_crf = MaskedCRF(
        use_mask=use_mask,
        label2idx_map=label2idx_map,
        num_output=num_output
    )

    run_config = RunConfig(
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoint_steps
    )
    model_fn = model_fn_builder(
        model_config=model_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        max_seq_length=max_seq_length,
        num_output=num_output,
        mask_crf=mask_crf
    )
    estimator = Estimator(
        model_fn=model_fn,
        params={
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size
        },
        config=run_config
    )
    return estimator


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_idx

    tf.logging.set_verbosity(tf.logging.INFO)

    model_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    data_processor = DataProcessor(
        train_file=FLAGS.train_file,
        dev_file=FLAGS.dev_file,
        test_file=FLAGS.test_file,
        max_seq_length=FLAGS.max_seq_length,
        vocab_file=FLAGS.vocab_file,
        data_format=FLAGS.data_format,
        do_lower_case=FLAGS.do_lower_case
    )

    output_dir = FLAGS.output_dir
    tf.gfile.MakeDirs(output_dir)

    # Loading data from input files
    train_examples = data_processor.train_examples
    num_train_steps = max(int(len(train_examples) / FLAGS.train_batch_size * FLAGS.epoch_num), FLAGS.min_train_steps)
    save_checkpoint_steps = int(num_train_steps / 10)

    tf.logging.info("***** Training Examples *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  Num save checkpoint steps = %d", save_checkpoint_steps)

    dev_examples = data_processor.dev_examples

    tf.logging.info("***** Dev Examples *****")
    tf.logging.info("  Num examples = %d", len(dev_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    test_examples = data_processor.test_examples

    tf.logging.info("***** Test Examples *****")
    tf.logging.info("  Num examples = %d", len(test_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # Writing data to data files with feature format
    train_file = os.path.join(output_dir, "train.tf_record")
    data_processor.file_based_convert_examples_to_features(
        examples=train_examples,
        output_file=train_file
    )

    dev_file = os.path.join(output_dir, "dev.tf_record")
    data_processor.file_based_convert_examples_to_features(
        examples=dev_examples,
        output_file=dev_file
    )

    test_file = os.path.join(output_dir, "test.tf_record")
    data_processor.file_based_convert_examples_to_features(
        examples=test_examples,
        output_file=test_file
    )

    # Building input_fn
    train_input_fn = data_processor.file_based_input_fn_builder(
        input_file=train_file,
        is_training=True
    )

    dev_input_fn = data_processor.file_based_input_fn_builder(
        input_file=dev_file,
        is_training=False
    )

    test_input_fn = data_processor.file_based_input_fn_builder(
        input_file=test_file,
        is_training=False
    )

    tf.logging.info("Label list: %s" % " ".join(list(data_processor.label2id_map.keys())))
    tf.logging.info("Label num: %d" % len(data_processor.label2id_map))

    # Building graph
    estimator = construct_estimator(
        output_dir=output_dir,
        save_checkpoint_steps=save_checkpoint_steps,
        model_config=model_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        lr_decay=FLAGS.lr_decay,
        max_seq_length=FLAGS.max_seq_length,
        use_mask=FLAGS.train_mask,
        label2idx_map=data_processor.label2id_map,
        num_output=len(data_processor.label2id_map),
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size
    )
    best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
    tf.gfile.MakeDirs(best_checkpoint_dir)

    best_ckpt_exporter = BestCheckpointsExporter(
        serving_input_receiver_fn=serving_fn,
        best_checkpoint_path=best_checkpoint_dir,
        compare_fn=_loss_smaller
    )

    # Constructing spec
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=num_train_steps,
    )
    test_spec = tf.estimator.EvalSpec(
        input_fn=dev_input_fn,
        steps=None,
        start_delay_secs=60,
        throttle_secs=60,
        exporters=best_ckpt_exporter
    )

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=test_spec
    )

    best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
    if os.path.exists(best_checkpoint_dir):
        with tf.gfile.GFile(os.path.join(best_checkpoint_dir, "best_checkpoint.txt"), "r") as fin:
            best_checkpoint = json.loads(fin.readline())["best_checkpoint_path"]
        fin.close()
    else:
        best_checkpoint = FLAGS.init_checkpoint

    tf.logging.info("***** Evaluate on the best checkpoint: %s *****" % best_checkpoint)

    best_estimator = construct_estimator(
        output_dir=best_checkpoint_dir,
        save_checkpoint_steps=save_checkpoint_steps,
        model_config=model_config,
        init_checkpoint=best_checkpoint,
        learning_rate=FLAGS.learning_rate,
        lr_decay=FLAGS.lr_decay,
        max_seq_length=FLAGS.max_seq_length,
        use_mask=FLAGS.eval_mask,
        label2idx_map=data_processor.label2id_map,
        num_output=len(data_processor.label2id_map),
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size
    )

    with tf.gfile.GFile(os.path.join(output_dir, "result_summary.txt"), "w") as writer:
        dev_results = best_estimator.predict(dev_input_fn, yield_single_examples=True)
        dev_metrics = evaluate(
            result=dev_results,
            id2label=data_processor.id2label_map,
            tokenizer=data_processor.tokenizer,
            predict_detail_file=os.path.join(output_dir, "dev_result.txt")
        )

        test_results = best_estimator.predict(test_input_fn, yield_single_examples=True)
        test_metrics = evaluate(
            result=test_results,
            id2label=data_processor.id2label_map,
            tokenizer=data_processor.tokenizer,
            predict_detail_file=os.path.join(output_dir, "test_result.txt")
        )

        writer.write("%s" % json.dumps({
            "dev": dev_metrics,
            "test": test_metrics
        }))

        tf.logging.info("***** Evaluate metrics *****")
        tf.logging.info("Dev: %s" % dev_metrics)
        tf.logging.info("Test: %s" % test_metrics)

    writer.close()


if __name__ == '__main__':
    tf.app.run()
