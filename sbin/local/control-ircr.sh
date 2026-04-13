#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
OUTPUT_DIR=$HOME/fs/$BASE/control/ircr/$TIMESTAMP
mkdir -p $OUTPUT_DIR

python $PARENT_DIR/src/$BASE/control/runner.py \
    --env MountainCarContinuous-v0 \
    --delay 3 \
    --num-steps 50000 \
    --reward-model-type ircr \
    --update-every-n-steps 1000 \
    --reward-model-kwarg max_buffer_size=200 \
    --reward-model-kwarg k_neighbors=5 \
    --sac-buffer-size 100000 \
    --sac-batch-size 256 \
    --sac-gradient-steps -1 \
    --log-episode-frequency 10 \
    --output-dir $OUTPUT_DIR \
    --seed 0
