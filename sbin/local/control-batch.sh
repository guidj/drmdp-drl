#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
OUTPUT_DIR=$HOME/fs/$BASE/control/drl-experiments/$TIMESTAMP
mkdir -p $OUTPUT_DIR

python $PARENT_DIR/src/$BASE/control/runner.py \
    --config-file $PARENT_DIR/specs/control-local-batch.json \
    --mode ray \
    --num-runs 3 \
    --update-every-n-steps 50 \
    --output-dir $OUTPUT_DIR

    
