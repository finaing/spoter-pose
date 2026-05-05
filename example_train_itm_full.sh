#!/usr/bin/bash -l
#SBATCH --job-name=itm_full
#SBATCH --time=6:0:0
#SBATCH --ntasks=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=%x_%j.out

# ── Paths ──────────────────────────────────────────────────────────────────────
# %x expands to --job-name above; change --job-name to rename everything at once.
EXPERIMENT_NAME="${SLURM_JOB_NAME}"

POSE_JSON="itm_data.json"
POSE_DIR="mediapipe_30_pose"          # directory containing <video_id>.pose files

# ── Training hyperparameters ───────────────────────────────────────────────────
EPOCHS=350
LR=0.001
HIDDEN_DIM=108                          # must equal num_landmarks * 2 = 54 * 2
SEED=379

# Scheduler
SCHEDULER_FACTOR=0.1
SCHEDULER_PATIENCE=5

# Gaussian noise (applied on top of augmentations during training)
GAUSSIAN_MEAN=0
GAUSSIAN_STD=0.001

# ── Environment ───────────────────────────────────────────────────────────────
module load miniforge3
source activate spoter-pose

#── Build the python command ───────────────────────────────────────────────────
CMD=(
    python train.py
    --experiment_name    "${EXPERIMENT_NAME}"
    --seed               ${SEED}
    --pose_json          "${POSE_JSON}"
    --pose_dir           "${POSE_DIR}"
    --validation_set     from-file
    --epochs             ${EPOCHS}
    --lr                 ${LR}
    --hidden_dim         ${HIDDEN_DIM}
    --scheduler_factor   ${SCHEDULER_FACTOR}
    --scheduler_patience ${SCHEDULER_PATIENCE}
    --gaussian_mean      ${GAUSSIAN_MEAN}
    --gaussian_std       ${GAUSSIAN_STD}
    --save_checkpoints   True
    --plot_stats         True
    --plot_lr            True
)

# ── Run ───────────────────────────────────────────────────────────────────────
"${CMD[@]}"
