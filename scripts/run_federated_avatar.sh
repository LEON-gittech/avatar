#!/bin/bash
###
 # @Author: LEON leon.kepler@bytedance.com
 # @Date: 2024-11-29 14:30:21
 # @LastEditors: LEON leon.kepler@bytedance.com
 # @LastEditTime: 2024-11-29 16:32:40
 # @FilePath: /ava/scripts/run_federated_avatar.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# Set environment variables
export PYTHONPATH="."

# Configuration
DATASET="amazon"
NUM_CLIENTS=1
N_ROUNDS=3
SPLIT_TYPE="random"  # or "group"
EMB_MODEL="text-embedding-ada-002"
AGENT_LLM="gpt-4"
API_FUNC_LLM="gpt-4"
OUTPUT_DIR="output/federated_avatar/${DATASET}"

# Run federated optimization
python scripts/run_federated_avatar.py \
    --dataset ${DATASET} \
    --num_clients ${NUM_CLIENTS} \
    --n_rounds ${N_ROUNDS} \
    --split_type ${SPLIT_TYPE} \
    --emb_model ${EMB_MODEL} \
    --agent_llm ${AGENT_LLM} \
    --api_func_llm ${API_FUNC_LLM} \
    --output_dir ${OUTPUT_DIR} \
    --model avatar \
    --aggregate llm 