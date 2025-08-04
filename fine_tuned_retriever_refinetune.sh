#!/bin/bash

#SBATCH --job-name=fine_tune_temporal
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -p suma_rtx4090  # 또는 사용 가능한 GPU 파티션

# 환경 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# conda 환경 활성화
source /home/bys3158/anaconda3/etc/profile.d/conda.sh
conda activate TimeR4

# numpy 버전 다운그레이드 (필요시)
# pip install numpy==1.23.5

# 실행 시간 기록
start_time=$(date +%s)
echo "Job started at $(date)"

# 파라미터 설정
DATA_PATH="datasets/MultiTQ/negatives_random1.json"

# 실행
python fine_tuned_retriever_refinetune.py --path $DATA_PATH --model_name /share0/bys3158/models/retriever/sentenseBERT_multitq_temporal

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job ended at $(date)"
echo "Total execution time: $duration seconds"
