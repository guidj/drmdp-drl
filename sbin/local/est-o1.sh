#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
python $PARENT_DIR/src/$BASE/dfdrl/est_o1.py \
        --delay=3 \
        --num-steps=100 \
        --train-epochs=100 \
        --output-dir=$HOME/fs/$BASE/dfdrl/$TIMESTAMP \
        --task-prefix $TIMESTAMP \
        --log-episode-frequency=5 \
        --use-seed
