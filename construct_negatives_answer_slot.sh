#!/bin/bash
#SBATCH --job-name=ver5_transrt2
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o slurm/slurm-%j.out
#SBATCH -d singleton
#SBATCH --mem=256G
#SBATCH -p suma_rtx4090

# 가상환경이 있다면 활성화 (필요한 경우 주석 해제)
source ~/anaconda3/bin/activate TimeR4

# 스크립트 실행
python construct_negatives_answer_slot.py

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo "✅ 스크립트가 성공적으로 실행되었습니다."
else
    echo "❌ 스크립트 실행 중 오류가 발생했습니다."
fi 