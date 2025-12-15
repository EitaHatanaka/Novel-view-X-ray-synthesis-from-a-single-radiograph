#!/bin/bash
#PBS -A XRAYDIFF
#PBS -q gpu
#PBS -l elapstim_req=00:30:00
#PBS -N zero123_test_10step_v2
#PBS -o /work/XRAYDIFF/eitahtnk/zero123-hf/logs_test/test_10step_v2.o$PBS_JOBID
#PBS -e /work/XRAYDIFF/eitahtnk/zero123-hf/logs_test/test_10step_v2.e$PBS_JOBID
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
# venv を有効化（あなたが作った環境を使用）
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

# set offline mode so wandb writes local run files and does not try to contact internet
export WANDB_MODE=offline
# where wandb will write run directories (must be on shared/persistent storage)
export WANDB_DIR=/work/XRAYDIFF/eitahtnk/zero123-hf/wandb_offline
# optional: project name used when later syncing
export WANDB_PROJECT=zero123_offline
# ---------------------------------------------
# 実行コマンド: 最大10ステップで終了させる (コメント文は削除)
# ---------------------------------------------
echo "[INFO] Starting test training for max 100 steps..."


python train_zero1to3.py \
  --train_data_dir /work/XRAYDIFF/share/lidcidri/img_simple_256 \
  --pretrained_model_name_or_path /work/XRAYDIFF/eitahtnk/zero123-hf/models \
  --output_dir logs_test \
  --train_batch_size 1 \
  --dataloader_num_workers 4 \
  --max_train_steps 10 \
  --report_to "wandb" \
  --enable_xformers_memory_efficient_attention False
echo "[INFO] JOB END"