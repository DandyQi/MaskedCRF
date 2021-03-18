#!/usr/bin/env bash

python train_and_eval.py \
    --bert_config_file=conf/bert_config.json \
    --train_file=data/atis/train.txt \
    --test_files=data/atis/dev.txt,data/atis/test.txt \
    --max_seq_length=128 \
    --vocab_file=conf/vocab.txt \
    --do_lower_case=True \
    --output_dir=model/tmp \
    --min_train_steps=1000 \
    --learning_rate=2e-5 \
    --do_train=True \
    --do_eval=False \
    --train_mask=True \
    --eval_mask=True \
    --lr_decay=1.0