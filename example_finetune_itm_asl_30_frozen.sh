#!/usr/bin/bash -l
#SBATCH --job-name=finetune_itm_asl_30_frozen
#SBATCH --time=2:0:0
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

# ── Transfer learning ──────────────────────────────────────────────────────────
# Leave PRETRAINED_MODEL empty to train from scratch.
# Set to a checkpoint path to fine-tune a pretrained encoder on this dataset.
PRETRAINED_MODEL="out-checkpoints/asl_citizen_pretrain_30/checkpoint_t_2.pth"

# Two-phase fine-tuning schedule (only used when PRETRAINED_MODEL is set):
#   Phase 1 (epochs 0 … FREEZE_EPOCHS-1): encoder frozen, only decoder + head trained.
#   Phase 2 (epochs FREEZE_EPOCHS … EPOCHS-1): full model trained at LR * FINETUNE_LR_FACTOR.
# Set FREEZE_EPOCHS=0 to skip phase 1 and fine-tune everything from the start.
FREEZE_EPOCHS=30
FINETUNE_LR_FACTOR=0.1

# ── Training hyperparameters ───────────────────────────────────────────────────
EPOCHS=150
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

if [[ -n "${PRETRAINED_MODEL}" ]]; then
    CMD+=(
        --pretrained_model   "${PRETRAINED_MODEL}"
        --freeze_encoder
        --freeze_epochs      ${FREEZE_EPOCHS}
        --finetune_lr_factor ${FINETUNE_LR_FACTOR}
    )
fi

# ── Run ───────────────────────────────────────────────────────────────────────
"${CMD[@]}"
