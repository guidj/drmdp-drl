#!/bin/bash
set -xe

# Use half the machine's capacity
ncpu=$(expr `nproc` / 2)
ncpu=$(($npcu>0 ? npcu : 1))
ray start --head --num-cpus=$ncpu
