#!/usr/bin/env bash
# -*- coding:utf-8 -*-
bash 'run_seq2seq_verbose.bash -d 0 -f myparser --label_smoothing 0 -l 1e-4 --lr_scheduler linear --warmup_steps 81 -b 4 --wo_constraint_decoding'