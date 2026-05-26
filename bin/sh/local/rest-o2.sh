#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../../..
BASE=drmdp

TIMESTAMP=`date +%s`
mkdir -p $HOME/fs/$BASE/dfdrl-rest/o1/$TIMESTAMP
ray job submit \
    --address http://127.0.0.1:8265 \
    --working-dir $PARENT_DIR \
    --runtime-env-json='{"py_modules":["src/drmdp"], "excludes": [".git"]}' \
    -- \
    python $PARENT_DIR/src/$BASE/dfdrl/est_o2.py \
        --min-delay 3 \
        --max-delay 5 \
        --buffer-num-steps=20000 \
        --train-epochs=1000 \
        --env "Hopper-v5" \
        --max-episode-steps 1000 \
        --output-dir=$HOME/fs/$BASE/dfdrl-rest/o2/$TIMESTAMP \
        --task-prefix $TIMESTAMP \
        --num-runs=1 \
        --log-episode-frequency=5 \
        --use-seed
