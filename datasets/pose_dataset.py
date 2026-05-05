"""
PoseFormatDataset – SPOTER dataset adapter for the pose-format (.pose) library.

Drop-in replacement for CzechSLRDataset that reads per-video .pose files
instead of a monolithic CSV with embedded landmark sequences.

Dependencies
------------
Install the pose-format library before use:
    pip install pose-format
"""

import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data
from random import randrange

from pose_format import Pose

from augmentations import (
    augment_arm_joint_rotate,
    augment_frame_dropout,
    augment_rotate,
    augment_shear,
)
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import HAND_IDENTIFIERS as _BASE_HAND_IDENTIFIERS
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

# Extend hand identifiers with left (_0) and right (_1) suffixes, matching
# the convention used in CzechSLRDataset and the augmentation utilities.
HAND_IDENTIFIERS = (
    [h + "_0" for h in _BASE_HAND_IDENTIFIERS]
    + [h + "_1" for h in _BASE_HAND_IDENTIFIERS]
)

# ── MediaPipe component names as stored in a Holistic .pose file ──────────────
_MP_POSE_COMPONENT       = "POSE_LANDMARKS"
_MP_LEFT_HAND_COMPONENT  = "LEFT_HAND_LANDMARKS"
_MP_RIGHT_HAND_COMPONENT = "RIGHT_HAND_LANDMARKS"

# ── Body landmark mapping: SPOTER name → MediaPipe POSE_LANDMARKS index ───────
# MediaPipe PoseLandmark (0-indexed):
#   0  NOSE                    11 LEFT_SHOULDER   12 RIGHT_SHOULDER
#   2  LEFT_EYE                 5 RIGHT_EYE
#   7  LEFT_EAR                 8 RIGHT_EAR
#  13  LEFT_ELBOW              14 RIGHT_ELBOW
#  15  LEFT_WRIST              16 RIGHT_WRIST
#
# "neck" has no MediaPipe equivalent; it is computed as the midpoint of
# LEFT_SHOULDER (11) and RIGHT_SHOULDER (12) – matching SPOTER's convention.
_MP_BODY_INDEX = {
    "nose":          0,
    "rightEye":      5,
    "leftEye":       2,
    "rightEar":      8,
    "leftEar":       7,
    "rightShoulder": 12,
    "leftShoulder":  11,
    "rightElbow":    14,
    "leftElbow":     13,
    "rightWrist":    16,
    "leftWrist":     15,
}
_MP_LEFT_SHOULDER_IDX  = 11
_MP_RIGHT_SHOULDER_IDX = 12

# ── SDPose component names ────────────────────────────────────────────────────
_SDP_POSE_COMPONENT       = "BODY"
_SDP_LEFT_HAND_COMPONENT  = "LEFT_HAND"
_SDP_RIGHT_HAND_COMPONENT = "RIGHT_HAND"

# ── Body landmark mapping: SPOTER name → SDPose BODY index ───────────────────
# SDPose BODY point order (0-indexed):
#   0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#   5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#   9 left_wrist, 10 right_wrist, 11–22 hips/legs/feet
#
# "neck" is not a named point; computed as midpoint of left_shoulder (5) and
# right_shoulder (6), same as MediaPipe handling.
_SDP_BODY_INDEX = {
    "nose":          0,
    "leftEye":       1,
    "rightEye":      2,
    "leftEar":       3,
    "rightEar":      4,
    "leftShoulder":  5,
    "rightShoulder": 6,
    "leftElbow":     7,
    "rightElbow":    8,
    "leftWrist":     9,
    "rightWrist":    10,
}
_SDP_LEFT_SHOULDER_IDX  = 5
_SDP_RIGHT_SHOULDER_IDX = 6

# ── AlphaPose component names as stored in a BODY_136 .pose file ──────────────
_AP_POSE_COMPONENT       = "BODY_136"
_AP_LEFT_HAND_COMPONENT  = "LEFT_HAND_136"
_AP_RIGHT_HAND_COMPONENT = "RIGHT_HAND_136"

# ── Body landmark mapping: SPOTER name → AlphaPose BODY_136 index ─────────────
# BODY_136 point order (0-indexed):
#  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
#  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
#  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle,
# 17 head_top, 18 neck, 19 pelvis, 20-25 feet/toes
#
# "neck" is a named point at index 18 — no midpoint computation needed.
_AP_BODY_INDEX = {
    "nose":          0,
    "leftEye":       1,
    "rightEye":      2,
    "leftEar":       3,
    "rightEar":      4,
    "leftShoulder":  5,
    "rightShoulder": 6,
    "leftElbow":     7,
    "rightElbow":    8,
    "leftWrist":     9,
    "rightWrist":    10,
    "neck":          18,
}

# ── Hand landmark mapping: SPOTER base name → MediaPipe HandLandmark index ────
# MediaPipe HandLandmark (0-indexed):
#   0 WRIST,  1 THUMB_CMC,  2 THUMB_MCP,  3 THUMB_IP,   4 THUMB_TIP,
#   5 INDEX_FINGER_MCP,  6 INDEX_FINGER_PIP,  7 INDEX_FINGER_DIP,  8 INDEX_FINGER_TIP,
#   9 MIDDLE_FINGER_MCP, 10 MIDDLE_FINGER_PIP, 11 MIDDLE_FINGER_DIP, 12 MIDDLE_FINGER_TIP,
#  13 RING_FINGER_MCP,  14 RING_FINGER_PIP,  15 RING_FINGER_DIP,  16 RING_FINGER_TIP,
#  17 PINKY_MCP,        18 PINKY_PIP,        19 PINKY_DIP,         20 PINKY_TIP
_HAND_MP_INDEX = {
    "wrist":     0,
    "indexTip":  8,
    "indexDIP":  7,
    "indexPIP":  6,
    "indexMCP":  5,
    "middleTip": 12,
    "middleDIP": 11,
    "middlePIP": 10,
    "middleMCP": 9,
    "ringTip":   16,
    "ringDIP":   15,
    "ringPIP":   14,
    "ringMCP":   13,
    "littleTip": 20,
    "littleDIP": 19,
    "littlePIP": 18,
    "littleMCP": 17,
    "thumbTip":  4,
    "thumbIP":   3,
    "thumbMP":   2,
    "thumbCMC":  1,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _component_xy(pose: Pose, component_name: str) -> np.ndarray:
    """
    Return a (Frames, Points, 2) float32 array for *component_name*,
    with masked / missing values filled with 0.

    Parameters
    ----------
    pose : Pose
        A loaded pose-format Pose object.
    component_name : str
        Name of the component to extract (e.g. "POSE_LANDMARKS").

    Returns
    -------
    np.ndarray, shape (Frames, Points, 2)
    """
    sub  = pose.get_components([component_name])
    data = sub.body.data  # (Frames, People, Points, Dims)

    # Masked numpy arrays: fill masked positions with 0 (convention used by SPOTER)
    if hasattr(data, "filled"):
        data = data.filled(0)

    data = np.asarray(data, dtype=np.float32)
    # Drop the 'people' axis (index 1) and keep only x, y (first 2 of Dims)
    return data[:, 0, :, :2]  # (Frames, Points, 2)


# ── Public API ────────────────────────────────────────────────────────────────

def load_pose_file(path: str) -> dict:
    """
    Load a .pose file and return a SPOTER-compatible landmark dictionary.

    Each value is a list of ``(x, y)`` tuples – one per frame – with
    coordinates normalised to [0, 1] by the image dimensions stored in the
    pose header.  This format is directly accepted by SPOTER's normalization
    and augmentation utilities.

    Both MediaPipe Holistic (``POSE_LANDMARKS`` / ``LEFT_HAND_LANDMARKS`` /
    ``RIGHT_HAND_LANDMARKS``) and AlphaPose 136 (``BODY_136`` /
    ``LEFT_HAND_136`` / ``RIGHT_HAND_136``) .pose files are supported; the
    format is detected automatically from the component names in the header.

    Parameters
    ----------
    path : str
        Path to the .pose file.

    Returns
    -------
    dict
        Keys: SPOTER landmark identifiers (body identifiers from
        ``normalization.body_normalization.BODY_IDENTIFIERS`` plus hand
        identifiers with ``_0`` / ``_1`` suffixes).
        Values: lists of ``(x, y)`` float tuples of length *num_frames*.
    """
    with open(path, "rb") as fh:
        pose = Pose.read(fh.read())

    component_names = {c.name for c in pose.header.components}

    if _MP_POSE_COMPONENT in component_names:
        return _load_mediapipe(pose)
    elif _AP_POSE_COMPONENT in component_names:
        return _load_alphapose(pose)
    elif _SDP_POSE_COMPONENT in component_names:
        return _load_sdpose(pose)
    else:
        raise ValueError(
            f"Unrecognised .pose format in {path}. "
            f"Components found: {component_names}"
        )


def _load_mediapipe(pose: Pose) -> dict:
    """Extract landmarks from a MediaPipe Holistic .pose file."""
    width  = float(pose.header.dimensions.width)
    height = float(pose.header.dimensions.height)

    body_data = _component_xy(pose, _MP_POSE_COMPONENT)        # (F, 33, 2)
    lh_data   = _component_xy(pose, _MP_LEFT_HAND_COMPONENT)   # (F, 21, 2)
    rh_data   = _component_xy(pose, _MP_RIGHT_HAND_COMPONENT)  # (F, 21, 2)

    num_frames = body_data.shape[0]
    result: dict = {}

    for spoter_name, idx in _MP_BODY_INDEX.items():
        result[spoter_name] = [
            (float(body_data[f, idx, 0] / width),
             float(body_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]

    # neck = midpoint of LEFT_SHOULDER and RIGHT_SHOULDER
    result["neck"] = [
        (
            float((body_data[f, _MP_LEFT_SHOULDER_IDX,  0] + body_data[f, _MP_RIGHT_SHOULDER_IDX, 0]) / (2.0 * width)),
            float((body_data[f, _MP_LEFT_SHOULDER_IDX,  1] + body_data[f, _MP_RIGHT_SHOULDER_IDX, 1]) / (2.0 * height)),
        )
        for f in range(num_frames)
    ]

    for base_name, idx in _HAND_MP_INDEX.items():
        result[base_name + "_0"] = [
            (float(lh_data[f, idx, 0] / width),
             float(lh_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]
        result[base_name + "_1"] = [
            (float(rh_data[f, idx, 0] / width),
             float(rh_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]

    return result


def _load_alphapose(pose: Pose) -> dict:
    """Extract landmarks from an AlphaPose 136-keypoint .pose file."""
    width  = float(pose.header.dimensions.width)
    height = float(pose.header.dimensions.height)

    body_data = _component_xy(pose, _AP_POSE_COMPONENT)        # (F, 26, 2)
    lh_data   = _component_xy(pose, _AP_LEFT_HAND_COMPONENT)   # (F, 21, 2)
    rh_data   = _component_xy(pose, _AP_RIGHT_HAND_COMPONENT)  # (F, 21, 2)

    num_frames = body_data.shape[0]
    result: dict = {}

    # All body landmarks including neck are direct index lookups.
    for spoter_name, idx in _AP_BODY_INDEX.items():
        result[spoter_name] = [
            (float(body_data[f, idx, 0] / width),
             float(body_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]

    # AlphaPose hand points (hand_0…hand_20) follow the same 21-point
    # MediaPipe ordering, so _HAND_MP_INDEX indices are reused as-is.
    for base_name, idx in _HAND_MP_INDEX.items():
        result[base_name + "_0"] = [
            (float(lh_data[f, idx, 0] / width),
             float(lh_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]
        result[base_name + "_1"] = [
            (float(rh_data[f, idx, 0] / width),
             float(rh_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]

    return result


def _load_sdpose(pose: Pose) -> dict:
    """Extract landmarks from an SDPose .pose file (BODY / LEFT_HAND / RIGHT_HAND)."""
    width  = float(pose.header.dimensions.width)
    height = float(pose.header.dimensions.height)

    body_data = _component_xy(pose, _SDP_POSE_COMPONENT)        # (F, 23, 2)
    lh_data   = _component_xy(pose, _SDP_LEFT_HAND_COMPONENT)   # (F, 21, 2)
    rh_data   = _component_xy(pose, _SDP_RIGHT_HAND_COMPONENT)  # (F, 21, 2)

    num_frames = body_data.shape[0]
    result: dict = {}

    for spoter_name, idx in _SDP_BODY_INDEX.items():
        result[spoter_name] = [
            (float(body_data[f, idx, 0] / width),
             float(body_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]

    # neck = midpoint of left_shoulder and right_shoulder (no dedicated point)
    result["neck"] = [
        (
            float((body_data[f, _SDP_LEFT_SHOULDER_IDX,  0] + body_data[f, _SDP_RIGHT_SHOULDER_IDX, 0]) / (2.0 * width)),
            float((body_data[f, _SDP_LEFT_SHOULDER_IDX,  1] + body_data[f, _SDP_RIGHT_SHOULDER_IDX, 1]) / (2.0 * height)),
        )
        for f in range(num_frames)
    ]

    # SDPose hand points follow MediaPipe 21-point ordering, so _HAND_MP_INDEX applies directly.
    for base_name, idx in _HAND_MP_INDEX.items():
        result[base_name + "_0"] = [
            (float(lh_data[f, idx, 0] / width),
             float(lh_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]
        result[base_name + "_1"] = [
            (float(rh_data[f, idx, 0] / width),
             float(rh_data[f, idx, 1] / height))
            for f in range(num_frames)
        ]

    return result


def landmark_dict_to_tensor(landmarks: dict) -> torch.Tensor:
    """
    Convert a SPOTER landmark dictionary to a float tensor of shape
    ``(Frames, 54, 2)``.

    The landmark order follows ``BODY_IDENTIFIERS + HAND_IDENTIFIERS``,
    matching the ordering expected by the SPOTER model.

    Parameters
    ----------
    landmarks : dict
        As returned by :func:`load_pose_file` (or after normalisation /
        augmentation).

    Returns
    -------
    torch.Tensor, shape (Frames, 54, 2)
    """
    all_ids    = BODY_IDENTIFIERS + HAND_IDENTIFIERS
    num_frames = len(landmarks[all_ids[0]])
    out = np.empty((num_frames, len(all_ids), 2), dtype=np.float32)

    for col, key in enumerate(all_ids):
        frames = landmarks[key]
        out[:, col, 0] = [fr[0] for fr in frames]
        out[:, col, 1] = [fr[1] for fr in frames]

    return torch.from_numpy(out)


class PoseFormatDataset(torch_data.Dataset):
    """
    PyTorch Dataset for sign language recognition that loads skeletal data
    from per-video .pose files (pose-format library).

    Produces the same ``(depth_map, label)`` output as ``CzechSLRDataset``
    and is therefore a drop-in replacement in ``train.py``.

    Do not instantiate directly; use one of the factory classmethods:

    * :meth:`from_csv`  – legacy SPOTER CSV format (1-indexed labels)
    * :meth:`from_json` – ``itm_data.json`` format (labels derived on the
      fly from ``word_label`` strings; dense and split-consistent)

    Parameters
    ----------
    pose_paths : list[str]
        Paths to .pose files, one per sample.
    labels : list[int]
        **0-indexed** class labels, one per sample.
    gloss_to_id : dict[str, int], optional
        Mapping from gloss string to integer label, as built by
        :meth:`from_json`.  Stored on the dataset for inspection and to
        derive ``num_classes`` without re-reading the JSON.
    transform : callable, optional
        Extra transform applied to the data tensor after all other processing.
    augmentations : bool, optional
        Enable random geometric augmentations.
    augmentations_prob : float, optional
        Probability of applying augmentation to any given sample.
    normalize : bool, optional
        Apply Bohacek body- and hand-normalisation (recommended).
    """

    def __init__(
        self,
        pose_paths: list,
        labels: list,
        gloss_to_id: dict = None,
        transform=None,
        augmentations: bool = False,
        augmentations_prob: float = 0.5,
        normalize: bool = True,
    ):
        self.pose_paths  = pose_paths
        self.labels      = labels
        self.targets     = list(labels)   # kept for compatibility with __balance_val_split
        self.gloss_to_id = gloss_to_id or {}
        self.transform   = transform
        self.augmentations      = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize   = normalize

    @property
    def num_classes(self) -> int:
        """Number of distinct classes, derived from ``gloss_to_id``."""
        return len(self.gloss_to_id)

    # ── Factory classmethods ──────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, csv_path: str, **kwargs) -> "PoseFormatDataset":
        """
        Load from a CSV file with ``pose_path`` and ``label`` columns.

        Labels in the CSV are expected to be **1-indexed** (SPOTER's original
        convention); they are converted to 0-indexed on load.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        **kwargs
            Forwarded to the dataset constructor (``transform``,
            ``augmentations``, ``augmentations_prob``, ``normalize``).
        """
        df = pd.read_csv(csv_path, encoding="utf-8")
        pose_paths = df["pose_path"].tolist()
        labels     = [lbl - 1 for lbl in df["label"].tolist()]   # 1-indexed → 0-indexed
        return cls(pose_paths, labels, **kwargs)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        pose_dir: str,
        split: str,
        **kwargs,
    ) -> "PoseFormatDataset":
        """
        Load from an ``itm_data.json`` metadata file.

        Labels are derived on the fly from the ``word_label`` field rather
        than taken from the pre-existing ``label`` integers.  The mapping is
        built from **all** entries in the JSON (not just the requested split)
        so that the label space is identical across train, val, and test.
        The resulting integer IDs are dense (0 to N-1) and stored in
        ``dataset.gloss_to_id``.

        Each entry's ``instances`` list is filtered to the requested split.
        The corresponding .pose file is expected at
        ``<pose_dir>/<video_id>.pose``.

        Parameters
        ----------
        json_path : str
            Path to the JSON metadata file (e.g. ``itm_data.json``).
        pose_dir : str
            Directory that contains the .pose files, one per ``video_id``.
        split : str
            Which split to load: ``"train"``, ``"val"``, or ``"test"``.
        **kwargs
            Forwarded to the dataset constructor (``transform``,
            ``augmentations``, ``augmentations_prob``, ``normalize``).
        """
        import json, os

        with open(json_path, encoding="utf-8") as fh:
            entries = json.load(fh)

        # Build a dense gloss → id mapping from ALL entries so the label
        # space is consistent regardless of which split is loaded.
        # Use dict.fromkeys to deduplicate while preserving order — duplicate
        # word_label entries in the JSON must map to the same id, not overwrite
        # each other with a higher index that would exceed num_classes.
        glosses     = [entry["word_label"] for entry in entries]
        gloss_to_id = {gloss: i for i, gloss in enumerate(dict.fromkeys(glosses))}

        pose_paths = []
        labels     = []
        missing    = 0

        for entry in entries:
            label = gloss_to_id[entry["word_label"]]
            for instance in entry["instances"]:
                if instance["split"] != split:
                    continue
                video_id  = instance["video_id"]
                pose_path = os.path.join(pose_dir, video_id + ".pose")
                if not os.path.isfile(pose_path):
                    missing += 1
                    continue
                pose_paths.append(pose_path)
                labels.append(label)

        if missing:
            logging.warning(
                "from_json(%s, split=%s): skipped %d missing .pose file(s).",
                json_path, split, missing,
            )

        return cls(pose_paths, labels, gloss_to_id=gloss_to_id, **kwargs)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Load, optionally augment and normalise a single sample.

        Returns
        -------
        depth_map : torch.Tensor, shape (Frames, 54, 2)
            Landmark coordinates shifted to the interval [-0.5, 0.5].
        label : torch.Tensor, shape (1,)
            0-indexed class label.
        """
        landmarks = load_pose_file(self.pose_paths[idx])
        label     = torch.Tensor([self.labels[idx]])   # already 0-indexed

        # ── Optional augmentations ────────────────────────────────────────────
        if self.augmentations and random.random() < self.augmentations_prob:
            aug = randrange(5)
            if aug == 0:
                landmarks = augment_rotate(landmarks, (-13, 13))
            elif aug == 1:
                landmarks = augment_shear(landmarks, "perspective", (0, 0.1))
            elif aug == 2:
                landmarks = augment_shear(landmarks, "squeeze", (0, 0.15))
            elif aug == 3:
                landmarks = augment_arm_joint_rotate(landmarks, 0.3, (-4, 4))
            elif aug == 4:
                landmarks = augment_frame_dropout(landmarks, dropout_min=0.0, dropout_max=0.3)

        # ── Optional Bohacek normalisation ────────────────────────────────────
        if self.normalize:
            landmarks = normalize_single_body_dict(landmarks)
            landmarks = normalize_single_hand_dict(landmarks)

        depth_map = landmark_dict_to_tensor(landmarks)
        depth_map = depth_map - 0.5   # shift to [-0.5, 0.5] as in CzechSLRDataset

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label
