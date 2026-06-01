#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../../..
BASE=drmdp

TIMESTAMP=`date +%s`
OUTPUT_PATH="$HOME/fs/$BASE/exp/rewest/$TIMESTAMP"

rm -rf $OUTPUT_PATH
mkdir $OUTPUT_PATH
ray job submit \
    --address http://127.0.0.1:8265 \
    --working-dir $PARENT_DIR \
    --runtime-env-json='{"py_modules":["src/drmdp"], "excludes": [".git"]}' \
    -- \
    python $PARENT_DIR/src/$BASE/workflows/rewest.py \
        --num-runs 1 \
        --num-episodes 500 \
        --output-path "$HOME/fs/$BASE/exp/rewest/$TIMESTAMP"
