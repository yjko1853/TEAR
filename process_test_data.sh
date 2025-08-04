#!/bin/bash

#SBATCH --job-name=ver5_transrt10
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o slurm/slurm-%j.out
#SBATCH -d singleton
#SBATCH --mem=256G
#SBATCH -p suma_rtx4090

# CUDA 관련 환경 변수 설정
# export CUDA_VISIBLE_DEVICES=0  # 첫 번째 GPU 사용

# Python 가상환경 활성화 (필요한 경우)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TimeR4

# 필요한 패키지 설치
# pip install transformers torch fuzzywuzzy python-Levenshtein tqdm

# 스크립트 실행
python process_test_data_timequestions.py 