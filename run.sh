#!/usr/bin/env bash

python train_and_eval.py \
    --bert_config_file=conf/bert_config.json \
    --train_file=data/resume/train.char.bio \
    --test_files=data/resume/dev.char.bio,data/resume/test.char.bio \
    --data_format=cols \
    --max_seq_length=128 \
    --vocab_file=conf/vocab.txt \
    --do_lower_case=True \
    --output_dir=model/resume \
    --min_train_steps=1000 \
    --learning_rate=2e-5 \
    --do_train=True \
    --do_eval=True \
    --train_mask=True \
    --eval_mask=True \
    --lr_decay=1.0