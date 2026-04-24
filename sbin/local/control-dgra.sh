#!/bin/bash
set -xe

DIR=$(dirname $0)
PARENT_DIR=$DIR/../..
BASE=drmdp

TIMESTAMP=`date +%s`
OUTPUT_DIR=$HOME/fs/$BASE/control/dgra/$TIMESTAMP
mkdir -p $OUTPUT_DIR

python $PARENT_DIR/src/$BASE/control/runner.py \
    --env MountainCarContinuous-v0 \
    --delay 3 \
    --num-steps 50000 \
    --reward-model-type dgra \
    --update-every-n-steps 1000 \
    --reward-model-kwarg max_buffer_size=5000 \
    --reward-model-kwarg train_epochs=100 \
    --reward-model-kwarg batch_size=64 \
    --reward-model-kwarg regu_lam=1.0 \
    --reward-model-kwarg learning_rate=1e-3 \
    --sac-buffer-size 100000 \
    --sac-batch-size 256 \
    --sac-gradient-steps -1 \
    --log-step-frequency 10 \
    --output-dir $OUTPUT_DIR \
    --seed 0
