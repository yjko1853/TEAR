#!/bin/bash

#SBATCH --job-name=main_original
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# 환경 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=4

# conda 환경 설정
source /home/bys3158/anaconda3/etc/profile.d/conda.sh
conda activate TimeR4

# 실행 시간 기록
echo "Job started at $(date)"

###################################
# Step1: Retrieve-rewrite-retrieve-rerank
###################################
echo "Running Step1: main_original.py with specified arguments..."
if [ -f "main_original.py" ]; then
    # Retriever 정보 출력
    # RETRIEVER_NAME="/share0/yjko1853/models/retriever/sentenseBERT_multitq"
    RETRIEVER_NAME="/share0/bys3158/models/retriever/sentenseBERT_multitq_temporal"
    # RETRIEVER_NAME="sentence-transformers/all-mpnet-base-v2"
    NUM_SUBGRAPHS=1000
    OUTPUT_FILENAME="question_quadruple_TimeR4_rewrite"
    QUESTION_PATH="datasets/MultiTQ/questions/test.json"
    TRIPLET_PATH="datasets/MultiTQ/kg/full.txt"
    OUTPUT_PATH="datasets/MultiTQ/results/test_prompt_timer4_rewrite_penalty_0.8_vote1_temporalBERT_wo_atc.json"
    ANSWER_SLOT_METHOD="penalty"  # masking, penalty, original 중 선택
    TEST_TEAR_PATH="datasets/MultiTQ/questions/test_tear_vote+entity.json"

    echo "============= 파라미터 정보 ============="
    echo "RETRIEVER_NAME: $RETRIEVER_NAME"
    echo "NUM_SUBGRAPHS: $NUM_SUBGRAPHS"
    echo "OUTPUT_FILENAME: $OUTPUT_FILENAME"
    echo "QUESTION_PATH: $QUESTION_PATH"
    echo "TRIPLET_PATH: $TRIPLET_PATH"
    echo "OUTPUT_PATH: $OUTPUT_PATH"
    echo "ANSWER_SLOT_METHOD: $ANSWER_SLOT_METHOD"
    echo "TEST_TEAR_PATH: $TEST_TEAR_PATH"
    echo "========================================="

    python main_original.py \
        --output_path $OUTPUT_PATH \
        --question_path $QUESTION_PATH \
        --retrieve_name $RETRIEVER_NAME \
        --triplet_path $TRIPLET_PATH \
        --num_subgraphs $NUM_SUBGRAPHS \
        --output_filename $OUTPUT_FILENAME \
        --answer_slot_method $ANSWER_SLOT_METHOD \
        --test_tear_path $TEST_TEAR_PATH
else
    echo "main_original.py not found. Please check your project structure."
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