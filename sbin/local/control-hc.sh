#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
OUTPUT_DIR=$HOME/fs/$BASE/control/hc/$TIMESTAMP
mkdir -p $OUTPUT_DIR

python $PARENT_DIR/src/$BASE/control/runner.py \
    --env MountainCarContinuous-v0 \
    --delay 3 \
    --num-steps 50000 \
    --agent-type hc \
    --agent-kwarg max_delay=3 \
    --agent-kwarg history_hidden_size=128 \
    --agent-kwarg reg_lambda=5.0 \
    --sac-buffer-size 100000 \
    --sac-batch-size 256 \
    --sac-gradient-steps -1 \
    --log-episode-frequency 10 \
    --output-dir $OUTPUT_DIR \
    --seed 0
