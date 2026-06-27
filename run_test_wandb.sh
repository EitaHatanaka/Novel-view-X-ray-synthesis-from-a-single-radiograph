#!/bin/bash
#PBS -A XRAYDIFF
#PBS -q gpu
#PBS -l elapstim_req=00:30:00
#PBS -N zero123_test_20step
#PBS -o /work/XRAYDIFF/eitahtnk/zero123-hf/logs_test/test_20step.o$PBS_JOBID
#PBS -e /work/XRAYDIFF/eitahtnk/zero123-hf/logs_test/test_20step.e$PBS_JOBID
#PBS -v OMP_NUM_THREADS=1


# ---------------------------------------------
# 初期設定
# ---------------------------------------------
cd $PBS_O_WORKDIR

echo "[INFO] JOB START"
# ... (中略)

# ---------------------------------------------
# モジュールロード（環境によって調整）
# ---------------------------------------------
module purge
module load intelpython/2022.3.1
module load cuda/12.1.0
module load openmpi/nvhpc-hpcx-cuda12/25.1

# ---------------------------------------------
# venv を有効化
# ---------------------------------------------
echo "[INFO] Activating virtual environment"
source /work/XRAYDIFF/eitahtnk/zero123-hf/zero123_env/bin/activate

# CUDA が見えているか確認
echo "[INFO] Python version: $(python --version)"
echo "[INFO] CUDA visible devices: $CUDA_VISIBLE_DEVICES"
python - << 'EOF'
import torch
print("[INFO] torch.cuda.is_available =", torch.cuda.is_available())
EOF

#wandbを使うために追加したブロック
export WANDB_MODE=offline
export WANDB_DIR=/work/XRAYDIFF/eitahtnk/zero123-hf/wandb_offline
export WANDB_PROJECT=zero123_offline

# ---------------------------------------------
# 実行コマンド (コメント文は削除)
# ---------------------------------------------
echo "[INFO] Starting test training for max 100 steps..."


python train_zero1to3.py \
  --train_data_dir /work/XRAYDIFF/share/lidcidri/img_simple_256 \
  --pretrained_model_name_or_path /work/XRAYDIFF/eitahtnk/zero123-165000 \
  --output_dir logs_test \
  --train_batch_size 2 \
  --dataloader_num_workers 4 \
  --max_train_steps 20 \
  --validation_steps 10 \
  --report_to "wandb" \
  --enable_xformers_memory_efficient_attention False
echo "[INFO] JOB END"
