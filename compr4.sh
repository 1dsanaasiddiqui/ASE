#!/usr/bin/env bash

source ./venv/bin/activate
echo "Activated venv"

function watch_mem {
    while ps -p $WATCHED_PID --no-headers --format "%mem" >> $LOG_FNAME; do
        sleep 0.1s
    done
}

python compress_net.py \
        -n networks/mnist/mnist_relu_3_100.onnx \
        -a 30 \
        -s 2000 \
        -b 1000 \
        -l "0.05, [ (100, 0.025), (10, 0.01), (1, 0.005) ]" \
        --force-no-end-relu True \
        -c 0
WATCHED_PID=$!
LOG_FNAME=logs/mem_4
echo launched
watch_mem
echo done
