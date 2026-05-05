"""
visualize_spoter_overlay.py
===========================
Render SPOTER-adapted pose landmarks either as a video overlay or on a plain
black background (pose-only mode).

Supports three estimators:
  - MediaPipe Holistic  (POSE_LANDMARKS / LEFT_HAND_LANDMARKS / RIGHT_HAND_LANDMARKS)
  - AlphaPose 136       (BODY_136 / LEFT_HAND_136 / RIGHT_HAND_136)
  - SDPose              (BODY / LEFT_HAND / RIGHT_HAND)

Only the subset of landmarks that SPOTER actually uses is drawn; all other
landmarks (lower body, face mesh, etc.) are suppressed by zeroing their
confidence so PoseVisualizer skips them.

Usage
-----
Overlay on video:

    python visualize_spoter_overlay.py \\
        --video <video.mp4> \\
        [--mp_pose <mediapipe.pose>] [--ap_pose <alphapose.pose>] \\
        [--sdp_pose <sdpose.pose>] \\
        [--out_mp <path>] [--out_ap <path>] [--out_sdp <path>]

Pose-only (black background, no video required):

    python visualize_spoter_overlay.py --pose-only \\
        [--mp_pose <mediapipe.pose>] [--ap_pose <alphapose.pose>] \\
        [--sdp_pose <sdpose.pose>] \\
        [--out_mp <path>] [--out_ap <path>] [--out_sdp <path>]

Dependencies
------------
    pip install pose-format opencv-python
    (vidgear and simple-video-utils are NOT required)
"""

import argparse
import math
import sys

# ── SPOTER landmark subsets ──────────────────────────────────────────────────

# MediaPipe POSE_LANDMARKS indices that SPOTER uses (0-indexed PoseLandmark):
#   0  NOSE   2  LEFT_EYE   5  RIGHT_EYE   7  LEFT_EAR   8  RIGHT_EAR
#  11  LEFT_SHOULDER  12  RIGHT_SHOULDER
#  13  LEFT_ELBOW     14  RIGHT_ELBOW
#  15  LEFT_WRIST     16  RIGHT_WRIST
_MP_BODY_KEEP = frozenset({0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16})

# AlphaPose BODY_136 indices that SPOTER uses:
#   0  nose   1  leftEye   2  rightEye   3  leftEar   4  rightEar
#   5  leftShoulder   6  rightShoulder
#   7  leftElbow      8  rightElbow
#   9  leftWrist      10 rightWrist
#  18  neck (named point, not a midpoint)
_AP_BODY_KEEP = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18})

# SDPose BODY indices that SPOTER uses (0-indexed):
#   0 nose  1 left_eye  2 right_eye  3 left_ear  4 right_ear
#   5 left_shoulder  6 right_shoulder
#   7 left_elbow     8 right_elbow
#   9 left_wrist    10 right_wrist
# Indices 11–22 (hips, legs, feet) are suppressed.
# neck is not a named point in SDPose; it is computed at load time in pose_dataset.py.
_SDP_BODY_KEEP = frozenset(range(11))

# All 21 hand landmarks are used by SPOTER for both left and right hands.
_HAND_KEEP = frozenset(range(21))

# Ear-to-shoulder skeleton edges are visually wrong for sign language (they form
# a large triangle across the torso).  Explicitly remove them per body component.
_BODY_LIMB_DENY = {
    # ear→shoulder (forms a distracting triangle across the torso)
    # left_eye↔right_eye (horizontal bar; M-shape without it is cleaner)
    "POSE_LANDMARKS": frozenset({(7, 11), (11, 7), (8, 12), (12, 8), (2, 5), (5, 2)}),
    "BODY_136":       frozenset({(3, 5),  (5, 3),  (4, 6),  (6, 4),  (1, 2), (2, 1)}),
    "BODY":           frozenset({(3, 5),  (5, 3),  (4, 6),  (6, 4),  (1, 2), (2, 1)}),
}


# ── Core helpers ─────────────────────────────────────────────────────────────

def _filter_to_spoter(pose):
    """
    Return a copy of *pose* in which only SPOTER-adapted landmarks are visible.

    Non-SPOTER points are suppressed by zeroing their confidence values.
    PoseVisualizer skips every point whose confidence is 0.

    Uses NumPyPoseBody.copy() (data.copy() / confidence.copy()) rather than
    copy.deepcopy, which is known to segfault on masked numpy arrays.
    """
    component_names = {c.name for c in pose.header.components}

    if "POSE_LANDMARKS" in component_names:
        body_comp = "POSE_LANDMARKS"
        lh_comp   = "LEFT_HAND_LANDMARKS"
        rh_comp   = "RIGHT_HAND_LANDMARKS"
        body_keep = _MP_BODY_KEEP
    elif "BODY_136" in component_names:
        body_comp = "BODY_136"
        lh_comp   = "LEFT_HAND_136"
        rh_comp   = "RIGHT_HAND_136"
        body_keep = _AP_BODY_KEEP
    elif "BODY" in component_names:
        body_comp = "BODY"
        lh_comp   = "LEFT_HAND"
        rh_comp   = "RIGHT_HAND"
        body_keep = _SDP_BODY_KEEP
    else:
        raise ValueError(
            f"Unrecognised .pose format. Components found: {component_names}"
        )

    pose_out = pose.copy()  # NumPyPoseBody.copy() → safe numpy array copies

    global_offset = 0
    for comp in pose_out.header.components:
        n = len(comp.points)

        if comp.name == body_comp:
            keep = body_keep
        elif comp.name in (lh_comp, rh_comp):
            keep = _HAND_KEEP
        else:
            keep = frozenset()  # suppress face mesh, lower body, etc.

        for local_i in range(n):
            if local_i not in keep:
                pose_out.body.confidence[:, :, global_offset + local_i] = 0.0

        # Remove skeleton edges where either endpoint is suppressed, plus any
        # explicitly denied pairs (e.g. ear→shoulder).
        # comp.colors are indexed by point index (mod len), not limb index — leave them alone.
        deny = _BODY_LIMB_DENY.get(comp.name, frozenset())
        comp.limbs = [
            (p1, p2) for p1, p2 in comp.limbs
            if p1 in keep and p2 in keep and (p1, p2) not in deny
        ]
        comp.relative_limbs = comp.get_relative_limbs()

        global_offset += n

    return pose_out


def _render(pose_filtered, output_path, video_path=None):
    """
    Render SPOTER-filtered pose to *output_path*.

    *video_path* is None  → pose-only mode: landmarks drawn on a black background
                            at the pose file's own FPS.
    *video_path* is given → overlay mode: landmarks drawn on top of video frames.
    """
    import cv2
    import numpy as np
    import os
    import shutil
    import subprocess
    import tempfile
    from pose_format.pose_visualizer import PoseVisualizer

    pose_fps = float(pose_filtered.body.fps)
    pose_w   = pose_filtered.header.dimensions.width
    pose_h   = pose_filtered.header.dimensions.height

    # ── open video (overlay mode only) ───────────────────────────────────────
    cap     = None
    out_fps = pose_fps
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isclose(out_fps, pose_fps, abs_tol=0.1):
            print(f"  Warning: video FPS ({out_fps:.3f}) != pose FPS ({pose_fps:.3f}). "
                  "Frames may drift.")

    # ── probe for a working OpenCV VideoWriter codec ──────────────────────────
    writer = None
    writer_path = output_path
    for out_path, fourcc_str in [
        (output_path,                            "avc1"),
        (output_path,                            "mp4v"),
        (output_path.rsplit(".", 1)[0] + ".avi", "MJPG"),
    ]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w = cv2.VideoWriter(out_path, fourcc, out_fps, (pose_w, pose_h))
        if w.isOpened():
            writer = w
            writer_path = out_path
            break

    use_ffmpeg = writer is None
    tmp_dir = None
    if use_ffmpeg:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "OpenCV has no video write backend and 'ffmpeg' is not on PATH. "
                "Install ffmpeg or a full opencv build."
            )
        tmp_dir = tempfile.mkdtemp(prefix="spoter_pose_")
        print(f"  No OpenCV video backend; writing frames to {tmp_dir}, then stitching with ffmpeg.")

    # ── build pose coordinate array (integer pixel space) ─────────────────────
    int_data   = np.array(np.around(pose_filtered.body.data.data), dtype="int32")
    confidence = pose_filtered.body.confidence
    num_pose_frames = len(int_data)

    vis = PoseVisualizer(pose_filtered)
    black_bg = np.zeros((pose_h, pose_w, 3), dtype=np.uint8)

    frame_idx = 0
    while frame_idx < num_pose_frames:
        if cap is not None:
            ret, bg_frame = cap.read()
            if not ret:
                break
            bg_frame = cv2.resize(bg_frame, (pose_w, pose_h))
        else:
            bg_frame = black_bg.copy()

        drawn = vis._draw_frame(
            int_data[frame_idx],
            confidence[frame_idx],
            bg_frame,
        )

        if use_ffmpeg:
            cv2.imwrite(os.path.join(tmp_dir, f"{frame_idx:06d}.jpg"), drawn)
        else:
            writer.write(drawn)
        frame_idx += 1

    if cap is not None:
        cap.release()

    if use_ffmpeg:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(out_fps),
                "-i", os.path.join(tmp_dir, "%06d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path,
            ],
            check=True,
        )
        shutil.rmtree(tmp_dir)
    else:
        writer.release()
        output_path = writer_path

    mode = "pose-only" if video_path is None else "overlay"
    print(f"  Wrote {frame_idx} frames [{mode}] → {output_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render SPOTER-adapted pose landmarks as a video overlay or on a "
            "plain black background (--pose-only)."
        )
    )
    parser.add_argument(
        "--video", default=None,
        help="Input MP4 video file (required for overlay mode; ignored with --pose-only).",
    )
    parser.add_argument("--mp_pose",  default=None, help="MediaPipe Holistic .pose file")
    parser.add_argument("--ap_pose",  default=None, help="AlphaPose 136-keypoint .pose file")
    parser.add_argument("--sdp_pose", default=None, help="SDPose .pose file")
    parser.add_argument(
        "--pose-only", action="store_true",
        help="Draw landmarks on a black background instead of a video frame.",
    )
    parser.add_argument(
        "--out_mp", default="out_mediapipe.mp4",
        help="Output path for MediaPipe render (default: out_mediapipe.mp4)",
    )
    parser.add_argument(
        "--out_ap", default="out_alphapose.mp4",
        help="Output path for AlphaPose render (default: out_alphapose.mp4)",
    )
    parser.add_argument(
        "--out_sdp", default="out_sdpose.mp4",
        help="Output path for SDPose render (default: out_sdpose.mp4)",
    )
    args = parser.parse_args()

    pose_only = args.pose_only or args.video is None

    if not pose_only and args.video is None:
        parser.error("Provide a video file, or pass --pose-only to render without video.")

    if not any([args.mp_pose, args.ap_pose, args.sdp_pose]):
        parser.error("Provide at least one pose file (mp_pose, ap_pose, or --sdp_pose).")

    try:
        from pose_format import Pose
    except ImportError:
        sys.exit(
            "pose-format is not installed.\n"
            "Run: pip install pose-format\n"
            "or:  pip install -e pose-master/src/python"
        )

    video_arg = None if pose_only else args.video

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    if args.mp_pose:
        print(f"Loading MediaPipe pose: {args.mp_pose}")
        with open(args.mp_pose, "rb") as fh:
            mp_pose = _filter_to_spoter(Pose.read(fh.read()))
        print(f"Rendering MediaPipe → {args.out_mp}")
        _render(mp_pose, args.out_mp, video_path=video_arg)

    # ── AlphaPose ─────────────────────────────────────────────────────────────
    if args.ap_pose:
        print(f"Loading AlphaPose pose: {args.ap_pose}")
        with open(args.ap_pose, "rb") as fh:
            ap_pose = _filter_to_spoter(Pose.read(fh.read()))
        print(f"Rendering AlphaPose → {args.out_ap}")
        _render(ap_pose, args.out_ap, video_path=video_arg)

    # ── SDPose ────────────────────────────────────────────────────────────────
    if args.sdp_pose:
        print(f"Loading SDPose pose: {args.sdp_pose}")
        with open(args.sdp_pose, "rb") as fh:
            sdp_pose = _filter_to_spoter(Pose.read(fh.read()))
        print(f"Rendering SDPose → {args.out_sdp}")
        _render(sdp_pose, args.out_sdp, video_path=video_arg)

    print("Done.")


if __name__ == "__main__":
    main()
