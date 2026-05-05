#!/usr/bin/bash -l
#SBATCH --job-name=test_ap
#SBATCH --time=1:0:0
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR="out-checkpoints/ap_double"

POSE_JSON="itm_data.json"
POSE_DIR="mp_poses"


# ── Environment ───────────────────────────────────────────────────────────────
module load miniforge3
source activate spoter-pose

# ── Run ───────────────────────────────────────────────────────────────────────
python test.py \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --pose_json       "${POSE_JSON}" \
    --pose_dir        "${POSE_DIR}"
