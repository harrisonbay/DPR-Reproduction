#!/bin/bash

MY_PYTHON="python3"
TRAINING_DATA='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-train.json'
DEV_DATA='/x0/arnavmd/nlp_proj/DPR/data/data/retriever/nq-dev.json'
BATCH=128
EPOCH=40
LR=1e-5
WORLD_SIZE=4
VERSION=0
WIKI='./wikipedia_data/data/wikipedia_split/psgs_w100_subset.tsv'
QA='nq-dev.csv'

cd ./src/
for k in {1,5}
do
  echo $VERSION
  $MY_PYTHON training.py --b $BATCH --e $EPOCH --lr $LR --train_set $TRAINING_DATA --dev_set $DEV_DATA --world_size $WORLD_SIZE --v $VERSION --model "BERT" --top_k $k
  $MY_PYTHON generate_embeddings.py --wiki $WIKI --qa_pair $QA --world_size $WORLD_SIZE --v $VERSION --b $BATCH
  $MY_PYTHON evaluate.py --wiki $WIKI --qa_pair $QA --world_size $WORLD_SIZE --v $VERSION
  ((VERSION++))
done

for m in "BERT" "DISTILBERT"
do
  echo $VERSION
  $MY_PYTHON training.py --b $BATCH --e $EPOCH --lr $LR --train_set $TRAINING_DATA --dev_set $DEV_DATA --world_size $WORLD_SIZE --v $VERSION --model $m --top_k 1
  $MY_PYTHON generate_embeddings.py --wiki $WIKI --qa_pair $QA --world_size $WORLD_SIZE --v $VERSION --b $BATCH
  $MY_PYTHON evaluate.py --wiki $WIKI --qa_pair $QA --world_size $WORLD_SIZE --v $VERSION
  ((VERSION++))
done
