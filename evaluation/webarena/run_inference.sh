#!/bin/bash
set -x

BASE_URL="http://ec2-3-84-138-66.compute-1.amazonaws.com"

export SHOPPING="$BASE_URL:7770/"
export SHOPPING_ADMIN="$BASE_URL:7780/admin"
export REDDIT="$BASE_URL:9999"
export GITLAB="$BASE_URL:8023"
export WIKIPEDIA="$BASE_URL:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export MAP="$BASE_URL:3000"
export HOMEPAGE="$BASE_URL:4399"

CURR_DIR=$PWD

cd ~
bash reset_webarena_host.sh
bash run_webarena_host.sh
sleep 180
cd $CURR_DIR
python main.py \
    --agent openhands \
    --output_dir evaluation/webarena/browsingagent-rand100 \
    --model gpt-4o \
    --dataset webarena \
    --config_name opendevin \
    --end_idx 100 \
    --shuffle \
    --seed 21 \
    --max_steps 15 

cd ~
bash reset_webarena_host.sh
bash run_webarena_host.sh
sleep 180
cd $CURR_DIR
python web_evaluation/webarena/inference_webarena.py \
    --agent reasoner \
    --output_dir evaluation/webarena/singlepolicy-rand100 \
    --model gpt-4o \
    --dataset webarena \
    --config_name browsergym_webarena \
    --end_idx 100 \
    --shuffle \
    --seed 21 \
    --max_steps 15 

cd ~
bash reset_webarena_host.sh
bash run_webarena_host.sh
sleep 180
cd $CURR_DIR
export AGENT_SELECTION="webarena_plan"
python web_evaluation/webarena/inference_webarena.py \
    --agent-cls reasoner \
    --output_dir web_evaluation/webarena/results/wmp-rand100 \
    --model gpt-4o \
    --dataset webarena \
    --config_name browsergym_webarena_world_model \
    --end_idx 100 \
    --shuffle \
    --seed 21 \
    --max_steps 15 
