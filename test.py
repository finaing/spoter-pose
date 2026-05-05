
import csv
import os
import re
import glob
import argparse
import logging
import torch

from torch.utils.data import DataLoader
from datasets.pose_dataset import PoseFormatDataset
from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.utils import evaluate, evaluate_top_k

_BEST_CKPT_RE = re.compile(r"The best checkpoint is (.+?)\.?\s*$")
_SLURM_ID_RE  = re.compile(r'_(\d+)$')


def get_args():
    parser = argparse.ArgumentParser()

    # ── dataset (mode A) ─────────────────────────────────────────────────────
    parser.add_argument("--pose_json", type=str, default="",
                        help="Path to itm_data.json; uses the 'test' split")
    parser.add_argument("--pose_dir", type=str, default="",
                        help="Directory containing .pose files (used with --pose_json)")
    parser.add_argument("--testing_set_path", type=str, default="",
                        help="Path to a CSV test set (pose_path,label)")
    parser.add_argument("--use_pose_format", action="store_true",
                        help="Use PoseFormatDataset for --testing_set_path")

    # ── mode A: sweep all checkpoints in one directory ────────────────────────
    parser.add_argument("--checkpoints_dir", type=str, default="",
                        help="Directory containing .pth checkpoint files")
    parser.add_argument("--top_k", type=int, nargs="+", default=None,
                        help="Report top-k accuracies, e.g. --top_k 1 3 5")

    # ── mode B: evaluate best checkpoint per training run ─────────────────────
    parser.add_argument("--eval_best", action="store_true",
                        help="Grep training_logs/*.out for best checkpoints and evaluate them")
    parser.add_argument("--logs_dir", type=str, default="training_logs",
                        help="Directory containing .out log files (default: training_logs)")
    parser.add_argument("--checkpoints_root", type=str, default="out-checkpoints",
                        help="Root directory for checkpoints (default: out-checkpoints)")
    parser.add_argument("--experiment_pattern", type=str, default="",
                        help="Regex to filter which experiments to process in --eval_best mode")
    parser.add_argument("--results_csv", type=str, default="",
                        help="Optional path to write results table as CSV")

    # ── per-estimator/size pose paths (auto-selected in --eval_best mode) ────
    parser.add_argument("--json_full",    type=str, default="", help="itm_data.json")
    parser.add_argument("--json_trimmed", type=str, default="", help="itm_data_trimmed.json")
    parser.add_argument("--json_minimal", type=str, default="", help="itm_data_minimal.json")
    parser.add_argument("--json_doubled", type=str, default="", help="itm_data_doubled.json")
    parser.add_argument("--dir_mp",         type=str, default="", help="MediaPipe pose dir")
    parser.add_argument("--dir_mp_doubled", type=str, default="", help="MediaPipe doubled pose dir")
    parser.add_argument("--dir_ap",         type=str, default="", help="AlphaPose pose dir")
    parser.add_argument("--dir_ap_doubled", type=str, default="", help="AlphaPose doubled pose dir")
    parser.add_argument("--dir_sdp",        type=str, default="", help="SDPose pose dir")

    return parser.parse_args()


def build_test_loader(args):
    if args.pose_json:
        test_set = PoseFormatDataset.from_json(args.pose_json, args.pose_dir, split="test")
    elif args.use_pose_format:
        test_set = PoseFormatDataset.from_csv(args.testing_set_path)
    else:
        test_set = CzechSLRDataset(args.testing_set_path)
    return DataLoader(test_set, shuffle=False)


def _resolve_dataset(folder: str, args):
    """
    Determine (pose_json, pose_dir) from the checkpoint folder name.
    Estimator: 'sdp' → SDPose, 'ap' → AlphaPose, else → MediaPipe.
    Size:      'double' / 'trimmed' / 'minimal' / (none) → full.
    Falls back to --pose_json / --pose_dir if a specific path is not set.
    """
    doubled  = 'double'  in folder
    trimmed  = 'trimmed' in folder
    minimal  = 'minimal' in folder

    if doubled:
        json = args.json_doubled or args.pose_json
    elif trimmed:
        json = args.json_trimmed or args.pose_json
    elif minimal:
        json = args.json_minimal or args.pose_json
    else:
        json = args.json_full    or args.pose_json

    if 'sdp' in folder:
        dir_ = args.dir_sdp or args.pose_dir
    elif 'ap' in folder:
        dir_ = (args.dir_ap_doubled if doubled else args.dir_ap) or args.pose_dir
    else:
        dir_ = (args.dir_mp_doubled if doubled else args.dir_mp) or args.pose_dir

    return json, dir_


def _make_loader(pose_json, pose_dir):
    dataset = PoseFormatDataset.from_json(pose_json, pose_dir, split="test")
    return DataLoader(dataset, shuffle=False)


def eval_best_checkpoints(args, device):
    """
    Scan training_logs/*.out for 'The best checkpoint is <folder>/<name>.',
    load out-checkpoints/<folder>/<name>.pth, evaluate top-1/3/5.

    When an experiment was submitted multiple times (different SLURM job IDs),
    only the run with the highest job ID is used.
    The dataset is selected automatically from the checkpoint folder name.
    """
    log_files = sorted(glob.glob(os.path.join(args.logs_dir, "*.out")))
    if not log_files:
        print(f"No .out files found in '{args.logs_dir}'")
        return

    pat = re.compile(args.experiment_pattern) if args.experiment_pattern else None

    # ── deduplicate: keep highest SLURM job ID per experiment name ────────────
    best_log = {}  # experiment → (job_id, log_path)
    for log_path in log_files:
        stem = os.path.splitext(os.path.basename(log_path))[0]
        m = _SLURM_ID_RE.search(stem)
        job_id     = int(m.group(1)) if m else 0
        experiment = stem[:m.start()] if m else stem

        if pat and not pat.search(experiment):
            continue

        if experiment not in best_log or job_id > best_log[experiment][0]:
            best_log[experiment] = (job_id, log_path)

    if not best_log:
        print("No matching experiments found.")
        return

    ks = [1, 3, 5]
    rows = []
    loader_cache = {}  # (pose_json, pose_dir) → DataLoader

    for experiment, (_, log_path) in sorted(best_log.items()):
        best_ckpt_rel = None
        with open(log_path, "r", errors="replace") as fh:
            for line in fh:
                m = _BEST_CKPT_RE.search(line)
                if m:
                    best_ckpt_rel = m.group(1).strip()

        if best_ckpt_rel is None:
            print(f"  [{experiment}] No best-checkpoint line found — skipping.")
            continue

        ckpt_path = os.path.join(args.checkpoints_root, best_ckpt_rel + ".pth")
        if not os.path.isfile(ckpt_path):
            print(f"  [{experiment}] Checkpoint not found: {ckpt_path} — skipping.")
            continue

        folder = best_ckpt_rel.split('/')[0]
        pose_json, pose_dir = _resolve_dataset(folder, args)
        if not pose_json or not pose_dir:
            print(f"  [{experiment}] Cannot resolve dataset for folder '{folder}' — skipping.")
            continue

        cache_key = (pose_json, pose_dir)
        if cache_key not in loader_cache:
            loader_cache[cache_key] = _make_loader(pose_json, pose_dir)
        test_loader = loader_cache[cache_key]

        print(f"  [{experiment}]  {folder}  →  {os.path.basename(pose_json)}")
        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.train(False)
        model.to(device)

        accs = {k: evaluate_top_k(model, test_loader, device, k=k)[2] for k in ks}
        rows.append({"experiment": experiment, **{f"top_{k}": accs[k] for k in ks}})

    if not rows:
        print("No results to report.")
        return

    # ── terminal table ─────────────────────────────────────────────────────────
    col_w = max(len(r["experiment"]) for r in rows)
    header = f"{'Experiment':<{col_w}}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}"
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        print(f"{r['experiment']:<{col_w}}  {r['top_1']:>7.4f}  {r['top_3']:>7.4f}  {r['top_5']:>7.4f}")
    print(sep)

    # ── optional CSV ───────────────────────────────────────────────────────────
    if args.results_csv:
        with open(args.results_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["experiment", "top_1", "top_3", "top_5"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.results_csv}")


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_best:
        eval_best_checkpoints(args, device)
        return

    # ── mode A: sweep all checkpoints in one directory ────────────────────────
    if not args.checkpoints_dir:
        print("Provide --checkpoints_dir or use --eval_best.")
        return

    test_loader = build_test_loader(args)

    checkpoints = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pth")))
    if not checkpoints:
        print(f"No .pth files found in '{args.checkpoints_dir}'")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in '{args.checkpoints_dir}'\n")

    ks = sorted(set(args.top_k)) if args.top_k else None
    top_acc, top_name = 0.0, ""

    for ckpt_path in checkpoints:
        name = os.path.basename(ckpt_path)
        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.train(False)
        model.to(device)

        if ks:
            accs = {k: evaluate_top_k(model, test_loader, device, k=k)[2] for k in ks}
            line = "  |  ".join(f"top-{k}: {accs[k]:.4f}" for k in ks)
            rank_acc = accs[ks[0]]
        else:
            _, _, rank_acc = evaluate(model, test_loader, device, print_stats=False)
            line = f"{rank_acc:.4f}"

        print(f"{name}  ->  {line}")
        logging.info("%s  ->  %s", name, line)

        if rank_acc > top_acc:
            top_acc = rank_acc
            top_name = name

    print(f"\nTop result: {top_acc:.4f}  ({top_name})")
    logging.info("Top result: %.4f  (%s)", top_acc, top_name)


if __name__ == "__main__":
    main()
