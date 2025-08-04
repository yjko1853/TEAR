#!/bin/bash

#SBATCH --job-name=main_time
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# 환경 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# conda 환경 설정
source /home/bys3158/anaconda3/etc/profile.d/conda.sh
conda activate TimeR4

# 실행 시간 기록
echo "Job started at $(date)"

###################################
# Step1: Retrieve-rewrite-retrieve-rerank
###################################
echo "Running Step1: main_timequestions.py with specified arguments..."
if [ -f "main_timequestions.py" ]; then
    # Retriever 정보 출력
    RETRIEVER_NAME="/share0/yjko1853/models/retriever/sentenseBERT_timequestions"
    # RETRIEVER_NAME="sentence-transformers/all-mpnet-base-v2"
    NUM_SUBGRAPHS=1000
    OUTPUT_FILENAME="question_quadruple_TimeR4_rewrite_timequestions"
    QUESTION_PATH="datasets/TimeQuestions/questions/test.json"
    TRIPLET_PATH="datasets/TimeQuestions/kg/full.txt"
    OUTPUT_PATH="datasets/TimeQuestions/results/test_prompt_timer4_rewrite_penalty_0.8_vote1_all.json"
    ANSWER_SLOT_METHOD="penalty"  # masking, penalty, original 중 선택
    ANSWER_SLOT_PATH="datasets/TimeQuestions/results/answer_slot_test_vote_1call_6samples_0.7.json"
    TOPIC_ENTITY_PATH="datasets/TimeQuestions/questions/test_tear_entity.json"

    echo "============= 파라미터 정보 ============="
    echo "RETRIEVER_NAME: $RETRIEVER_NAME"
    echo "NUM_SUBGRAPHS: $NUM_SUBGRAPHS"
    echo "OUTPUT_FILENAME: $OUTPUT_FILENAME"
    echo "QUESTION_PATH: $QUESTION_PATH"
    echo "TRIPLET_PATH: $TRIPLET_PATH"
    echo "OUTPUT_PATH: $OUTPUT_PATH"
    echo "ANSWER_SLOT_METHOD: $ANSWER_SLOT_METHOD"
    echo "ANSWER_SLOT_PATH: $ANSWER_SLOT_PATH"
    echo "TOPIC_ENTITY_PATH: $TOPIC_ENTITY_PATH"
    echo "========================================="

    python main_timequestions.py \
        --output_path $OUTPUT_PATH \
        --question_path $QUESTION_PATH \
        --retrieve_name $RETRIEVER_NAME \
        --triplet_path $TRIPLET_PATH \
        --num_subgraphs $NUM_SUBGRAPHS \
        --output_filename $OUTPUT_FILENAME \
        --answer_slot_method $ANSWER_SLOT_METHOD \
        --answer_slot_path $ANSWER_SLOT_PATH \
        --topic_entity_path $TOPIC_ENTITY_PATH
else
    echo "main_timequestions.py not found. Please check your project structure."
    exit 1
fi

###################################
# Step2: Reasoning
###################################
# echo "Running Step2: predict_answer.py with specified arguments..."
# if [ -f "predict_answer.py" ]; then
#     python predict_answer.py \
#         --model_path /share0/yjko/models/LLMs/Llama2_MultiTQ2/checkpoint-1815 \
#         -d datasets/MultiTQ/results/test_prompt.json \
#         --debug \
#         --predict_path datasets/MultiTQ/result
# else
#     echo "predict_answer.py not found. Please check your project structure."
#     exit 1
# fi

# 실행 시간 기록
echo "Job ended at $(date)"

# 실행 시간 계산
start_seconds=$(date -d "$start_time" +%s)
end_seconds=$(date -d "$end_time" +%s)
duration=$((end_seconds - start_seconds))

# 시간을 일, 시간, 분, 초로 변환
days=$((duration / 86400))
hours=$(( (duration % 86400) / 3600 ))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Total execution time: ${days}d ${hours}h ${minutes}m ${seconds}s"

# Notify completion
echo "Pipeline execution completed."

##SBATCH -p suma_a6000
#SBATCH -p suma_rtx4090