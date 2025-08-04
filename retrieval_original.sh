#!/bin/bash

#SBATCH --job-name=original_retrieval
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o slurm2/slurm-%j.out
#SBATCH -d singleton
#SBATCH --mem=256G
#SBATCH -p suma_rtx4090

# 시작 시간 기록
start_time=$(date '+%Y-%m-%d %H:%M:%S')
echo "Job started at: $start_time"

# Activate conda environment
source ~/anaconda3/bin/activate TimeR4

# 하이퍼파라미터 설정
TOP_K=20                    # 추출할 상위 쿼드러플 개수
USE_SUBGRAPH=false         # 서브그래프 사용 여부 (true/false)
SUBGRAPH_PATH="datasets/MultiTQ/results/question_quadruple_answer_slot_200.pt"  # 서브그래프 파일 경로
TRANSRT_WEIGHT=0.0         # TransRT 가중치 (0: SentenceBERT만, 1: TransRT만)
TRANSRT_MODEL="saved_models/transrt_model_v1_23.pt"  # TransRT 모델 경로
SBERT_MODEL="/share0/yjko1853/models/retriever/sentenseBERT_multitq"  # Fine-tuned SentenceBERT 모델 경로

# 하이퍼파라미터 출력
echo "=== 하이퍼파라미터 설정 ==="
echo "TOP_K: $TOP_K"
echo "USE_SUBGRAPH: $USE_SUBGRAPH"
echo "SUBGRAPH_PATH: $SUBGRAPH_PATH"
echo "TRANSRT_WEIGHT: $TRANSRT_WEIGHT"
echo "TRANSRT_MODEL: $TRANSRT_MODEL"
echo "SBERT_MODEL: $SBERT_MODEL"
echo "=========================="

# 명령어 구성
CMD="python retrieval_original.py \
    --top_k $TOP_K \
    --transrt_weight $TRANSRT_WEIGHT \
    --transrt_model_path $TRANSRT_MODEL \
    --sbert_model_path $SBERT_MODEL"

# 서브그래프 사용 여부에 따라 명령어 추가
if [ "$USE_SUBGRAPH" = true ]; then
    CMD="$CMD --use_subgraph --subgraph_path $SUBGRAPH_PATH"
fi

# 명령어 출력 및 실행
echo -e "\n실행할 명령어:"
echo $CMD
echo -e "\n=== 실행 시작 ===\n"
eval $CMD

# 종료 시간 기록 및 실행 시간 계산
end_time=$(date '+%Y-%m-%d %H:%M:%S')
echo -e "\nJob ended at: $end_time"

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
echo "Original retrieval execution completed." 