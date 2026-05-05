
import os
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils import __balance_val_split, __split_of_train_sequence, __log_class_statistics
from datasets.czech_slr_dataset import CzechSLRDataset
from datasets.pose_dataset import PoseFormatDataset
from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate
from spoter.gaussian_noise import GaussianNoise


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=64, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--use_pose_format", action="store_true",
                        help="Load data from .pose files via PoseFormatDataset instead of the legacy CSV format. "
                             "The CSV must have 'pose_path' and 'label' columns.")
    parser.add_argument("--pose_json", type=str, default="",
                        help="Path to an itm_data.json metadata file. When set, --pose_dir must also be provided "
                             "and data is loaded via PoseFormatDataset.from_json instead of a CSV.")
    parser.add_argument("--pose_dir", type=str, default="",
                        help="Directory containing .pose files named <video_id>.pose, used together with --pose_json.")
    # Transfer learning
    parser.add_argument("--pretrained_model", type=str, default="",
                        help="Path to a pretrained SPOTER checkpoint (.pth). When set, all encoder weights are "
                             "transferred and only the classification head is re-initialised for --num_classes.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the transformer encoder (and positional parameters) at the start of training. "
                             "Use together with --freeze_epochs for a two-phase schedule.")
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="Number of epochs to keep the encoder frozen before unfreezing for end-to-end "
                             "fine-tuning. Only effective when --freeze_encoder is set. "
                             "0 = start fully unfrozen (end-to-end from epoch 1).")
    parser.add_argument("--finetune_lr_factor", type=float, default=0.1,
                        help="LR multiplier applied when the encoder is unfrozen after --freeze_epochs. "
                             "Keeps the pretrained encoder stable during end-to-end fine-tuning.")

    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--no_scheduler", action="store_true",
                        help="Disable the ReduceLROnPlateau scheduler and train at a fixed LR (matches the original SPOTER paper)")
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Loss
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor for CrossEntropyLoss (0 = off). "
                             "Values around 0.1 help when training on noisy or pseudo-labels.")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=float, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=float, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    return parser


def train(args):

    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + ("_" + str(args.experimental_train_split).replace(".", "") if args.experimental_train_split else "") + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)


    # MARK: DATA

    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])

    if args.pose_json:
        # ── JSON metadata mode (itm_data.json) ───────────────────────────────
        # Splits are encoded inside the JSON; a single file covers train/val/test.
        train_set = PoseFormatDataset.from_json(
            args.pose_json, args.pose_dir, split="train",
            transform=transform, augmentations=True,
        )

        # Derive num_classes before any split replaces train_set with a Subset.
        args.num_classes = train_set.num_classes

        # Validation set
        if args.validation_set == "from-file":
            val_set = PoseFormatDataset.from_json(args.pose_json, args.pose_dir, split="val")
            val_loader = DataLoader(val_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)
        elif args.validation_set == "split-from-train":
            train_set, val_set = __balance_val_split(train_set, 0.2)
            val_set.dataset.transform = None
            val_set.dataset.augmentations = False
            val_loader = DataLoader(val_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)
        else:
            val_loader = None

        # Testing set
        eval_set    = PoseFormatDataset.from_json(args.pose_json, args.pose_dir, split="test")
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True) \
            if len(eval_set) > 0 else None

    elif args.use_pose_format:
        # ── CSV mode with .pose files ─────────────────────────────────────────
        train_set = PoseFormatDataset.from_csv(
            args.training_set_path, transform=transform, augmentations=True,
        )

        if args.validation_set == "from-file":
            val_set = PoseFormatDataset.from_csv(args.validation_set_path)
            val_loader = DataLoader(val_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)
        elif args.validation_set == "split-from-train":
            train_set, val_set = __balance_val_split(train_set, 0.2)
            val_set.dataset.transform = None
            val_set.dataset.augmentations = False
            val_loader = DataLoader(val_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)
        else:
            val_loader = None

        eval_loader = None
        if args.testing_set_path:
            eval_set    = PoseFormatDataset.from_csv(args.testing_set_path)
            eval_loader = DataLoader(eval_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)

    else:
        # ── Legacy CSV mode (original SPOTER) ─────────────────────────────────
        train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)

        if args.validation_set == "from-file":
            val_set = CzechSLRDataset(args.validation_set_path)
            val_loader = DataLoader(val_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)
        elif args.validation_set == "split-from-train":
            train_set, val_set = __balance_val_split(train_set, 0.2)
            val_set.dataset.transform = None
            val_set.dataset.augmentations = False
            val_loader = DataLoader(val_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)
        else:
            val_loader = None

        eval_loader = None
        if args.testing_set_path:
            eval_set    = CzechSLRDataset(args.testing_set_path)
            eval_loader = DataLoader(eval_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g, num_workers=4, persistent_workers=True)

    # MARK: MODEL
    # Constructed here so that args.num_classes is already updated from the
    # dataset (JSON mode sets it automatically from the gloss vocabulary).

    if args.pretrained_model:
        # Only freeze on load when freeze_epochs > 0; freeze_epochs=0 means
        # train fully end-to-end from the start (freeze for 0 epochs).
        slrt_model = SPOTER.from_pretrained(
            args.pretrained_model,
            num_classes=args.num_classes,
            freeze_encoder=args.freeze_encoder and args.freeze_epochs > 0,
        )
    else:
        slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)

    slrt_model.train(True)
    slrt_model.to(device)

    cel_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    sgd_optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, slrt_model.parameters()), lr=args.lr
    )
    scheduler = None if args.no_scheduler else optim.lr_scheduler.ReduceLROnPlateau(
        sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(args.epochs):

        # ── Two-phase fine-tuning: unfreeze encoder after freeze_epochs ───────
        if (args.pretrained_model and args.freeze_encoder
                and args.freeze_epochs > 0 and epoch == args.freeze_epochs):
            slrt_model.unfreeze_encoder()
            finetune_lr = args.lr * args.finetune_lr_factor
            sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=finetune_lr)
            scheduler = None if args.no_scheduler else optim.lr_scheduler.ReduceLROnPlateau(
                sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience
            )
            print(f"Epoch {epoch + 1}: encoder unfrozen, LR reset to {finetune_lr:.2e}")
            logging.info("Epoch %d: encoder unfrozen, LR reset to %.2e", epoch + 1, finetune_lr)

        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, device)
        losses.append(train_loss.item() / len(train_loader))
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            _, _, val_acc = evaluate(slrt_model, val_loader, device)
            slrt_model.train(True)
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                checkpoint_path = "out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth"
                if not os.path.exists(checkpoint_path):
                    continue
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load(checkpoint_path, weights_only=False)
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")


    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)
