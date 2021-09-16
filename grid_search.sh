#!/usr/bin/env bash

for lr in '5e-5' '2e-5' '1e-5'
do
    for decay in '1.0' '0.75' '0.5'
    do
        for i in 0 1 2
        do
            python train_and_eval.py \
                --bert_config_file=conf/bert_config.json \
                --init_checkpoint=model/roberta/bert_model.ckpt \
                --train_file=data/resume/train.char.bio \
                --dev_file=data/resume/dev.char.bio \
                --test_file=data/resume/test.char.bio \
                --data_format=bio \
                --max_seq_length=128 \
                --epoch_num=3 \
                --train_batch_size=32 \
                --vocab_file=conf/vocab.txt \
                --do_lower_case=True \
                --output_dir=model/resume-grid-search/lr-${lr}-decay-${decay}-ex-${i} \
                --min_train_steps=10000 \
                --learning_rate=${lr} \
                --train_mask=True \
                --eval_mask=True \
                --lr_decay=${decay} \
                --gpu_idx=3

        done
    done
done
