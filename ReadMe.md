# Masked-CRF

Masked Conditional Random Field (Masked-CRF) is an easy to implement variant CRF that impose restrictions on candidate paths during both training and decoding phases.

For more detail, refer to our paper:


# Train and Evaluate
This example code fine-tunes Bert + Masked-CRF on ATIS data sets.

`train_mask`: whether to use Masked-CRF during training

`eval_mask`: whether to use Masked-CRF during evaluating

`lr_decay`: control the learning rate for each layer

```bash
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
    --do_eval=True \
    --train_mask=True \
    --eval_mask=True \
    --lr_decay=1.0
```

# Performance

|  Task   |  Bert-CRF  |  Bert-MCRF-Decoding  |  Bert-MCRF-Training  |
|  ----  | ----  | ---- | ---- |
| Resume  |  |  |  |
| MSRA  |  |  |  |
| Ontonotes  |  |  |  |
| Weibo  |  |  |  |
| ATIS  |  |  |  |
| SNIPS  |  |  |  |
| CoNLL2000 |  |  |  |
| CoNLL2003 |  |  |  |