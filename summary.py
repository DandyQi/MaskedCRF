#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 下午3:06
# @Author  : qijianwei
# @File    : summary.py.py
# @Usage:


import os
import json
import numpy as np


def summary(output_dir, lr_lst, n, decay_lst):
    dev_results, test_results = [], []
    for lr in lr_lst:
        for decay in decay_lst:
            for i in range(n):
                result_file = os.path.join(
                    output_dir,
                    "lr-%s-decay-%s-ex-%s" % (lr, decay, i),
                    "result_summary.txt"
                )
                if not os.path.exists(result_file):
                    continue
                data = json.load(open(result_file, "r"))
                dev_results.append(round(data["dev"]["f1"], 2))
                test_results.append(round(data["test"]["f1"], 2))
                print("%s" % json.dumps({
                    "lr": lr,
                    "decay": decay,
                    "ex": i,
                    "dev": dev_results[-1],
                    "test": test_results[-1]
                }))

    dev_max_value = max(dev_results)
    dev_max_idx = np.argmax(dev_results).item()
    test_max_value = max(test_results)
    max_dev_test = test_results[dev_max_idx]

    print("Max dev: %s" % dev_max_value)
    print("Max test: %s" % test_max_value)
    print("Max dev's test: %s" % max_dev_test)


if __name__ == '__main__':
    summary(
        "model/resume-grid-search",
        lr_lst=["5e-5", "2e-5", "1e-5"],
        decay_lst=["1.0", "0.75", "0.5"],
        n=3
    )

