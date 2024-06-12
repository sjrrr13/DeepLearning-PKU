#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../experiment/exp1
PRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2    # will download pre-trained CodeGPT model
LOGFILE=../experiment/exp1/gpt2_train.log

python run.py \
    --data_dir=$DATADIR \
    --langs=$LANG \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --model_type=gpt2 \
    --block_size=512 \
    --do_train \
    --learning_rate=5e-5 \
    --weight_decay=0.01 \
    --evaluate_during_training \
    --per_gpu_train_batch_size=6 \
    --per_gpu_eval_batch_size=12 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=30 \
    --logging_steps=100 \
    --save_steps=5000 \
    --overwrite_output_dir \
    --seed=2024

