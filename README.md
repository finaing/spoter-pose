# SPOTER + pose-format

> Adaptation of [SPOTER](https://github.com/matyasbohacek/spoter) by **[Matyáš Boháček](https://github.com/matyasbohacek)** and **[Marek Hrúz](https://github.com/mhruz)**, University of West Bohemia.
> This adaptation was developed at the University of Zurich.

This repository extends SPOTER to load skeletal data directly from [pose-format](https://github.com/sign-language-processing/pose) `.pose` files rather than from pre-flattened CSV landmark sequences. It adds support for three pose estimators — **MediaPipe Holistic**, **AlphaPose 136**, and **SDPose** — and introduces transfer learning, an enhanced test runner, and SLURM scripts for HPC cluster training.

The SPOTER model architecture, augmentations, normalization, and training logic are unchanged from the original. See the [original repository](https://github.com/matyasbohacek/spoter) and the [paper](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html) for full details.

---

## Principal Differences from the Original SPOTER

### 1. Pose-format input (`datasets/pose_dataset.py`)

The original SPOTER reads landmark coordinates from a CSV file where each row is a video and each column is a flattened `(x, y)` coordinate for a specific landmark and frame. This adaptation replaces that with `PoseFormatDataset`, a drop-in replacement for `CzechSLRDataset` that reads per-video `.pose` files using the [pose-format](https://github.com/sign-language-processing/pose) library.

All three supported estimators are automatically detected from the component names in the `.pose` header and mapped to the same SPOTER landmark dictionary (`BODY_IDENTIFIERS + HAND_IDENTIFIERS`, shape `(Frames, 54, 2)`):

| Estimator | Body component | Hand components | Neck |
|---|---|---|---|
| MediaPipe Holistic | `POSE_LANDMARKS` | `LEFT/RIGHT_HAND_LANDMARKS` | midpoint of shoulders |
| AlphaPose 136 | `BODY_136` | `LEFT/RIGHT_HAND_136` | named point at index 18 |
| SDPose | `BODY` | `LEFT/RIGHT_HAND` | midpoint of shoulders |

All three produce identical tensor format after loading; the estimator difference is fully encapsulated in `load_pose_file()`.

### 2. JSON metadata loading

`PoseFormatDataset.from_json()` loads directly from a JSON metadata file instead of a CSV. The JSON is a flat list of gloss entries; each entry groups all video instances of one sign together with a string label (`word_label`) and a split assignment per instance. Integer class IDs are derived on the fly from the full list of `word_label` values — preserving insertion order and deduplicating — so the label space is identical whether you load `"train"`, `"val"`, or `"test"`.

The `.pose` file for each instance is looked up as `<pose_dir>/<video_id>.pose`.

**Example of JSON structure:**

```json
[
  {
    "word_label": "hello",
    "instances": [
      { "video_id": "hello_001", "split": "train" },
      { "video_id": "hello_002", "split": "train" },
      { "video_id": "hello_003", "split": "val"   },
      { "video_id": "hello_004", "split": "test"  }
    ]
  },
  {
    "word_label": "goodbye",
    "instances": [
      { "video_id": "goodbye_001", "split": "train" },
      { "video_id": "goodbye_002", "split": "test"  }
    ]
  }
]
```

Only `word_label`, `instances[*].video_id`, and `instances[*].split` are read; any additional fields in the file are ignored.

This setup was used for an Icelandic Sign Language (ÍTM) dataset that had word labels instead of glosses, but the code can easily be adapted to match a different key (such as `gloss`).

### 3. Transfer learning (`train.py`)

Three new arguments support two-phase fine-tuning from a pretrained SPOTER checkpoint:

- `--pretrained_model` — path to a `.pth` checkpoint; encoder weights are transferred and the classification head is re-initialised
- `--freeze_encoder` / `--freeze_epochs` — freeze the transformer encoder for the first N epochs, then unfreeze for end-to-end fine-tuning
- `--finetune_lr_factor` — LR multiplier applied when the encoder is unfrozen

### 4. Enhanced test runner (`test.py`)

The original `test.py` evaluates a single checkpoint. This version supports two modes:

- **Mode A** (`--checkpoints_dir`): sweeps all `.pth` files in a directory, optionally reporting top-k accuracies (`--top_k 1 3 5`)
- **Mode B** (`--eval_best`): greps SLURM `.out` log files for the best checkpoint per training run, resolves the correct pose estimator and dataset size from the experiment name, and writes a results CSV

### 5. Visualization (`scripts/visualize_spoter_overlay.py`)

Overlays the SPOTER-adapted subset of pose landmarks (body + both hands; lower body and face mesh suppressed) on an MP4 video for all three estimators. Uses the native `PoseVisualizer` from pose-format with the skeleton connections and colors stored in each `.pose` file's header.

---

## Installation

```shell
pip install -r requirements.txt
```

---

## Training

### From a pose-format JSON + directory of `.pose` files

```shell
python train.py \
  --experiment_name my_experiment \
  --epochs 100 \
  --lr 0.001 \
  --pose_json /path/to/itm_data.json \
  --pose_dir  /path/to/pose_files/
```

### From a CSV with `pose_path` and `label` columns

```shell
python train.py \
  --experiment_name my_experiment \
  --use_pose_format \
  --training_set_path   train.csv \
  --validation_set_path val.csv \
  --testing_set_path    test.csv
```

### From the original SPOTER CSV format (unchanged)

```shell
python train.py \
  --experiment_name my_experiment \
  --training_set_path   train.csv \
  --validation_set_path val.csv \
  --testing_set_path    test.csv
```

### Transfer learning from a pretrained checkpoint

```shell
python train.py \
  --experiment_name finetune \
  --pretrained_model pretrained.pth \
  --freeze_encoder \
  --freeze_epochs 20 \
  --finetune_lr_factor 0.1 \
  --pose_json /path/to/itm_data.json \
  --pose_dir  /path/to/pose_files/
```

All other hyperparameters (`--hidden_dim`, `--gaussian_std`, `--label_smoothing`, `--scheduler_*`, etc.) are documented in `train.py`.

---

## Testing

### Evaluate a single checkpoint

```shell
python test.py \
  --pose_json /path/to/itm_data.json \
  --pose_dir  /path/to/pose_files/ \
  --checkpoints_dir out-checkpoints/my_experiment/ \
  --top_k 1 3 5
```

### Sweep best checkpoints across multiple training runs

```shell
python test.py \
  --eval_best \
  --logs_dir         training_logs/ \
  --checkpoints_root out-checkpoints/ \
  --results_csv      results.csv \
  --json_full        /path/to/itm_data.json \
  --dir_mp           /path/to/mediapipe_poses/ \
  --dir_ap           /path/to/alphapose_poses/ \
  --dir_sdp          /path/to/sdpose_poses/
```

---

## Example SLURM Scripts

The four `example_*.sh` scripts in are self-contained SLURM job files that were used in experiments with the ÍTM dataset. Each sets its own hyperparameters and paths at the top; to reuse one, copy it, update the path variables, and submit with `sbatch`.

### `example_train_itm_full.sh` — train from scratch on ÍTM

Trains a fresh SPOTER model on the full ÍTM dataset (MediaPipe poses, 30 fps) for 350 epochs at lr 0.001 with a patience-5 ReduceLROnPlateau scheduler.

```shell
sbatch example_train_itm_full.sh
```

Key settings: `POSE_JSON=itm_data.json`, `POSE_DIR=mediapipe_30_pose`, `EPOCHS=350`, `HIDDEN_DIM=108`.

### `example_pretrain_asl_citizen.sh` — pretrain on ASL Citizen

Trains on the ASL Citizen dataset for 30 epochs to produce a cross-lingual source checkpoint. `PRETRAINED_MODEL` is empty by default (train from scratch); set it to a `.pth` path to fine-tune instead. The two-phase schedule variables (`FREEZE_EPOCHS`, `FINETUNE_LR_FACTOR`) are wired up but dormant until `PRETRAINED_MODEL` is set.

```shell
sbatch example_pretrain_asl_citizen.sh
```

Key settings: `POSE_JSON=asl_citizen_data.json`, `POSE_DIR=ASL_Citizen/poses`, `EPOCHS=30`.

### `example_finetune_itm_asl_30_frozen.sh` — fine-tune ÍTM from ASL Citizen checkpoint

Loads the ASL Citizen checkpoint produced by the previous script and fine-tunes on ÍTM using a two-phase schedule: encoder frozen for the first 30 epochs (only the classification head trains), then unfrozen for the remaining 120 epochs at `lr × 0.1`.

```shell
sbatch example_finetune_itm_asl_30_frozen.sh
```

Key settings: `PRETRAINED_MODEL=out-checkpoints/asl_citizen_pretrain_30/checkpoint_t_2.pth`, `FREEZE_EPOCHS=30`, `FINETUNE_LR_FACTOR=0.1`, `EPOCHS=150`.

### `example_test.sh` — evaluate all checkpoints in a directory

Runs `test.py` (Mode A) over every `.pth` file in `CHECKPOINTS_DIR` and reports top-1 accuracy for each. Edit `CHECKPOINTS_DIR`, `POSE_JSON`, and `POSE_DIR` to point at a different experiment.

```shell
sbatch example_test.sh
```

Key settings: `CHECKPOINTS_DIR=out-checkpoints/ap_double`, `POSE_JSON=itm_data.json`.

---

## Visualization

Overlay SPOTER-adapted landmarks on a video for comparison across estimators:

```shell
python scripts/visualize_spoter_overlay.py \
  video.mp4 mediapipe.pose alphapose.pose \
  --sdp_pose sdpose.pose \
  --out_mp overlay_mp.mp4 \
  --out_ap overlay_ap.mp4 \
  --out_sdp overlay_sdp.mp4
```

---

## License

This adaptation is published under the same [Apache License 2.0](LICENSE) as the original SPOTER codebase.

The **code** may be used for both academic and commercial purposes provided that the License and copyright notice are included, the original work is cited, and all changes are stated.

© 2022 Matyáš Boháček and Marek Hrúz (original SPOTER)

---

## Citation

If you use this work, please cite the original SPOTER paper:

```bibtex
@InProceedings{Bohacek_2022_WACV,
    author    = {Boh\'a\v{c}ek, Maty\'a\v{s} and Hr\'uz, Marek},
    title     = {Sign Pose-Based Transformer for Word-Level Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {182-191}
}
```
