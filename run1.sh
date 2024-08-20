#!/usr/bin/env bash

source ./venv/bin/activate
echo "Activated venv"

function watch_mem {
    while ps -p $WATCHED_PID --no-headers --format "%mem" >> $LOG_FNAME; do
        sleep 0.1s
    done
}

echo "With 15 abs nodes"
python parametrized_abstract_network.py \
        --network ../networks/ACASXU_run2a_1_3_batch_2000.nnet \
        --property ../properties/property_3.prop \
        --abs-nodes 15 \
        --epochs 100 &
WATCHED_PID=$!
LOG_FNAME=logs/mem_15_abs_nodes
echo launched
watch_mem
echo done
