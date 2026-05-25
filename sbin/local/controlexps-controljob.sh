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
    python $PARENT_DIR/src/$BASE/workflows/controljob.py \
        --num-runs=1 \
        --num-episodes=100 \
        --output-dir=$HOME/fs/$BASE/workflows/controljob/logs/$TIMESTAMP \
        --task-prefix $TIMESTAMP \
        --log-step-frequency=5 \
        --use-seed
