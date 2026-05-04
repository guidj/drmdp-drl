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
    --sac-kwarg buffer_size=100000 \
    --sac-kwarg batch_size=256 \
    --sac-kwarg gradient_steps=-1 \
    --sac-kwarg ent_coef=auto_0.1 \
    --output-dir $OUTPUT_DIR \
    --seed 0
