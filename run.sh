#!/usr/bin/env bash

echo df -kh
df -kh
echo du -sh $2/*
df -kh

chemprop_train --data_path "$2" --dataset_type classification --save_dir tox21_checkpoints