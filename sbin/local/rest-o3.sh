#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
mkdir -p $HOME/fs/$BASE/dfdrl-rest/o3/$TIMESTAMP
ray job submit \
    --address http://127.0.0.1:8265 \
    --working-dir $PARENT_DIR \
    --runtime-env-json='{"py_modules":["src/drmdp"], "excludes": [".git"]}' \
    -- \
    python $PARENT_DIR/src/$BASE/dfdrl/est_o3.py \
        --delay=3 \
        --buffer-num-steps=100 \
        --train-epochs=10 \
        --output-dir=$HOME/fs/$BASE/dfdrl-rest/o3/$TIMESTAMP \
        --num-runs=3 \
        --log-episode-frequency=5
