#!/bin/bash
set -xe

JOB_ID=$1
ray job stop $JOB_ID --address http://127.0.0.1:8265/
