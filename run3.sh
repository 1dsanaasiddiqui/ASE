#!/usr/bin/env bash

source ./venv/bin/activate
echo "Activated venv"

function watch_mem {
    while ps -p $WATCHED_PID --no-headers --format "%mem" >> $LOG_FNAME; do
        sleep 0.1s
    done
}

echo "With 75 abs nodes no lam fix"
python parametrized_abstract_network.py \
        --network ../networks/ACASXU_run2a_1_3_batch_2000.nnet \
        --property ../properties/property_3.prop \
        --abs-nodes 75 \
        --epochs 100 \
        --lambda-fix-enable False &
WATCHED_PID=$!
LOG_FNAME=logs/mem_75_abs_nodes_no_lam_fix
echo launched
watch_mem
echo done
