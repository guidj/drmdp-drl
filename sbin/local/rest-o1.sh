#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
ray job submit \
    --address http://127.0.0.1:8265 \
    --working-dir $PARENT_DIR \
    --runtime-env-json='{"py_modules":["src/drmdp"], "excludes": [".git"]}' \
    -- \
    python $PARENT_DIR/src/$BASE/dfdrl/est_o1.py \
        --delay=3 \
        --num-steps=100 \
        --train-epochs=100 \
        --output-dir=$HOME/fs/$BASE/dfdrl-rest/o1/$TIMESTAMP \
        --task-prefix $TIMESTAMP \
        --num-runs=3 \
        --log-episode-frequency=5 \
        --use-seed
