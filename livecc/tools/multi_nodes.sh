#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MASTER_PORT=${MASTER_PORT:-29501}
export GPUS=${GPUS:-8}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}
export RUN_CMD=${RUN_CMD:-"${SCRIPT_DIR}/sft_single_node.sh"}

# IP list: the first host is master, the rest are workers.
# Keep "127.0.0.1" for single-node runs, or add worker IPs for multi-node training.
IP_LIST=(
"127.0.0.1"
)

NNODES=${#IP_LIST[@]}
export NNODES

MASTER=${IP_LIST[0]}
export MASTER

echo "=========================================="
echo "Multi-Node Training Launch Script"
echo "=========================================="
echo "Master: ${MASTER}"
echo "Workers: ${IP_LIST[@]:1}"
echo "Total Nodes: ${NNODES}"
echo "GPUs per Node: ${GPUS}"
echo "Master Port: ${MASTER_PORT}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "=========================================="
echo ""

###########################################
# Step 1: start all worker nodes in the background
###########################################
for i in $(seq 1 $((NNODES - 1))); do
    WORKER=${IP_LIST[$i]}
    NODE_RANK=$i

    echo ">>> Launching worker ${i} (rank=${NODE_RANK}) on ${WORKER} ..."

    ssh ${WORKER} "
        set -e
        export MASTER=${MASTER}
        export MASTER_PORT=${MASTER_PORT}
        export NNODES=${NNODES}
        export GPUS=${GPUS}
        export LEARNING_RATE=${LEARNING_RATE}
        export NODE_RANK=${NODE_RANK}
        nohup bash ${RUN_CMD} > /dev/null 2>&1 &
    " &

    echo "   Worker ${i} launch command sent"
done

echo ">>> Waiting for workers to initialize..."

###########################################
# Step 2: start the master node in the foreground and save logs
###########################################
echo ""
echo ">>> Launching master (rank=0) on ${MASTER} ..."
echo ">>> Logs will be saved on master node"
echo ""

export NODE_RANK=0

bash ${RUN_CMD} 2>&1 | tee "${SCRIPT_DIR}/master.log"
echo ""
echo ">>> Training completed on master"
