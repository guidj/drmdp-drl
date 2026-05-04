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
    --agent-kwarg history_hidden_size=128 \
    --agent-kwarg reg_lambda=5.0 \
    --sac-kwarg buffer_size=100000 \
    --sac-kwarg batch_size=256 \
    --sac-kwarg gradient_steps=-1 \
    --sac-kwarg ent_coef=auto_0.1 \
    --output-dir $OUTPUT_DIR \
    --seed 0
