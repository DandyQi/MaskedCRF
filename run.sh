#!/usr/bin/env bash

python train_and_eval.py \
    --bert_config_file=conf/bert_config.json \
    --train_file=data/resume/train.char.bio \
    --dev_file=data/resume/dev.char.bio \
    --test_file=data/resume/test.char.bio \
    --data_format=bio \
    --max_seq_length=128 \
    --epoch_num=1 \
    --train_batch_size = 32 \
    --vocab_file=conf/vocab.txt \
    --do_lower_case=True \
    --output_dir=model/resume \
    --min_train_steps=1000 \
    --learning_rate=2e-5 \
    --train_mask=True \
    --eval_mask=True \
    --lr_decay=1.0 \
    --gpu_idx=0