#!/bin/bash
#PBS -A HAIRDESC
#PBS -q gen_S
#PBS -l elapstim_req=0:05:00
#PBS -N zero123_infer_finetuned
#PBS -o /work/XRAYDIFF/eitahtnk/zero123-hf/logs/infer_finetuned.standard.o
#PBS -e /work/XRAYDIFF/eitahtnk/zero123-hf/logs/infer_finetuned.error.e
#PBS -v OMP_NUM_THREADS=1

cd $PBS_O_WORKDIR
echo "[INFO] JOB START (Inference)"

module purge
module load intelpython/2022.3.1
module load cuda/12.1.0
module load openmpi/nvhpc-hpcx-cuda12/25.1

echo "[INFO] Activating virtual environment"
source /work/XRAYDIFF/eitahtnk/zero123-hf/zero123_env/bin/activate

# ★ pipeline_zero1to3.py を import できるようにする
export PYTHONPATH=/work/XRAYDIFF/eitahtnk/zero123-hf:$PYTHONPATH

echo "[INFO] Python version: $(python --version)"
echo "[INFO] CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python - << 'EOF'
import torch, sys
print("[INFO] torch version =", torch.__version__)
print("[INFO] torch.cuda.is_available =", torch.cuda.is_available())
print("[INFO] torch.cuda.device_count =", torch.cuda.device_count())
print("[INFO] sys.path[0:3] =", sys.path[0:3])
EOF

echo "[INFO] Starting inference..."

# ★ここが変更点：学習後に保存した final_save_dir を model_path に渡す
python /work/XRAYDIFF/eitahtnk/zero123-hf/svdrr-wrist/inference_GIF.py \
  
echo "[INFO] JOB END (Inference)"
