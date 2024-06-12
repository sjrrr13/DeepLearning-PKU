#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../test/ckpt6
LOGFILE=../test/ckpt6/test_gpt2.log
PRETRAINDIR=../experiment/exp1/checkpoint-30000-2.0195

python -u run.py \
    --data_dir=$DATADIR \
    --langs=$LANG \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --model_type=gpt2 \
    --block_size=512 \
    --do_infer \
    --logging_steps=100 \
    --seed=2024