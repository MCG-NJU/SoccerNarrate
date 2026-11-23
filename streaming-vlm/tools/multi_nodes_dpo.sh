#!/bin/bash

set -e

#################### Basic config ####################
export MASTER_PORT=${MASTER_PORT:-29501}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export BETA=${BETA:-0.1}
export LENGTH_NORM=${LENGTH_NORM:-True}

# Data and model settings forwarded to tools/dpo.sh.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${PROJECT_DIR}/.." && pwd)

export DATASET_PATH=${DATASET_PATH:-"${REPO_ROOT}/data"}
export PRETRAINED_MODEL_NAME_OR_PATH=${PRETRAINED_MODEL_NAME_OR_PATH:-"mit-han-lab/StreamingVLM"}
export REF_MODEL_NAME_OR_PATH=${REF_MODEL_NAME_OR_PATH:-""} # optional
export INCLUDE_PREFIX_VIDEO=${INCLUDE_PREFIX_VIDEO:-True}

LOG_DIR="${SCRIPT_DIR}"

# Repo path on each node. Export REPO_DIR manually if paths differ across nodes.
export REPO_DIR=${REPO_DIR:-"${PROJECT_DIR}"}

# Single-node torchrun training script.
export RUN_CMD="${REPO_DIR}/tools/dpo.sh"

# IP list. The first host is the master and the rest are workers.
IP_LIST=(
"127.0.0.1"
)
# Add more hosts here when running on multiple nodes.

#################### Compute distributed parameters ####################
NNODES=${#IP_LIST[@]}
export NNODES

MASTER=${IP_LIST[0]}
export MASTER
export MASTER_ADDR=${MASTER}
export GPUS=${GPUS_PER_NODE}

echo "=========================================="
echo "Multi-Node Training Launch Script (torchrun)"
echo "=========================================="
echo "Master: ${MASTER}"
echo "Workers: ${IP_LIST[@]:1}"
echo "Total Nodes: ${NNODES}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "Master Port: ${MASTER_PORT}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "BETA: ${BETA}"
echo "LENGTH_NORM: ${LENGTH_NORM}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "REPO_DIR: ${REPO_DIR}"
echo "=========================================="
echo ""

#################### Launch worker nodes ####################
# node_rank follows the IP_LIST order: 0 is master, the rest are workers.
echo ">>> Launching torchrun job on workers via SSH ..."

for idx in "${!IP_LIST[@]}"; do
    NODE_IP=${IP_LIST[$idx]}

    # The master starts locally below; start only workers here.
    if [ "$NODE_IP" != "$MASTER" ]; then
        echo "  - Worker node ${NODE_IP}, node_rank=${idx}"
        ssh "${NODE_IP}" "
            set -e
            export MASTER_ADDR=${MASTER_ADDR}
            export MASTER_PORT=${MASTER_PORT}
            export NNODES=${NNODES}
            export GPUS_PER_NODE=${GPUS_PER_NODE}
            export NODE_RANK=${idx}
            export LEARNING_RATE=${LEARNING_RATE}
            export BETA=${BETA}
            export LENGTH_NORM=${LENGTH_NORM}
            export DATASET_PATH='${DATASET_PATH}'
            export PRETRAINED_MODEL_NAME_OR_PATH='${PRETRAINED_MODEL_NAME_OR_PATH}'
            export REF_MODEL_NAME_OR_PATH='${REF_MODEL_NAME_OR_PATH}'
            export INCLUDE_PREFIX_VIDEO='${INCLUDE_PREFIX_VIDEO}'
            export REPO_DIR='${REPO_DIR}'
            cd '${REPO_DIR}'
            bash -lc '${RUN_CMD}'" \
        2>&1 | tee "${LOG_DIR}/worker_${idx}.log" &
    fi
done

#################### Launch master ####################
echo ""
echo ">>> Launching torchrun job on master ..."

export NODE_RANK=0
cd "${REPO_DIR}"

bash "${RUN_CMD}" 2>&1 | tee "${LOG_DIR}/master.log"

echo ""
echo ">>> Master finished, waiting for workers to exit ..."
wait

echo ">>> All nodes exited, training finished."

# Optional post-run hook.
if [[ -n "${KEEP_GPUS_BUSY_SCRIPT:-}" && -f "${KEEP_GPUS_BUSY_SCRIPT}" ]]; then
    bash "${KEEP_GPUS_BUSY_SCRIPT}"
fi
