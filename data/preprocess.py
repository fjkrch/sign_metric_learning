"""
Preprocessing pipeline: extract MediaPipe hand landmarks from video files,
normalise (translation + scale), and save as NumPy arrays.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    mp_hands = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def extract_landmarks_from_video(
    video_path: str,
    max_frames: int = 32,
    hands_model=None,
) -> Optional[np.ndarray]:
    """Extract 21 3-D hand landmarks per frame from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Fixed sequence length (pad/truncate).
        hands_model: Reusable MediaPipe Hands instance.

    Returns:
        Array of shape ``(max_frames, 21, 3)`` or ``None`` on failure.
    """
    if not _HAS_MEDIAPIPE:
        raise ImportError("mediapipe and opencv-python are required for video preprocessing. "
                          "Install with: pip install mediapipe opencv-python")
    own_model = hands_model is None
    if own_model:
        hands_model = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    landmarks_seq: List[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_model.process(rgb)
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = np.array([[p.x, p.y, p.z] for p in hand.landmark])  # (21, 3)
            landmarks_seq.append(lm)
        else:
            # Use zeros as placeholder when no hand detected
            landmarks_seq.append(np.zeros((21, 3), dtype=np.float32))

    cap.release()
    if own_model:
        hands_model.close()

    if len(landmarks_seq) == 0:
        return None

    arr = np.stack(landmarks_seq, axis=0).astype(np.float32)  # (T, 21, 3)

    # Truncate or pad to max_frames
    if arr.shape[0] >= max_frames:
        arr = arr[:max_frames]
    else:
        pad = np.zeros((max_frames - arr.shape[0], 21, 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    return arr


def normalize_landmarks(
    landmarks: np.ndarray,
    translate: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Normalise hand landmarks to be translation- and scale-invariant.

    Rigid-transform invariance property:
        If x' = Rx + t  then  ||x'_i - x'_j|| == ||x_i - x_j||
    Wrist-centring removes translation *t*; dividing by hand span removes
    the effect of distance to camera.

    Args:
        landmarks: Array of shape ``(T, 21, 3)``.
        translate: Centre landmarks on the wrist (landmark 0).
        scale: Divide by max pairwise distance (hand span).

    Returns:
        Normalised landmarks array of the same shape.
    """
    lm = landmarks.copy()
    T = lm.shape[0]

    for t in range(T):
        frame = lm[t]
        if np.allclose(frame, 0):
            continue  # skip zero-padded frames

        # Translation: wrist-centred
        if translate:
            wrist = frame[0:1]  # (1, 3)
            frame = frame - wrist

        # Scale: divide by hand span (max pairwise distance)
        if scale:
            diffs = frame[:, None, :] - frame[None, :, :]  # (21, 21, 3)
            dists = np.linalg.norm(diffs, axis=-1)          # (21, 21)
            max_dist = dists.max()
            if max_dist > 1e-6:
                frame = frame / max_dist

        lm[t] = frame
    return lm


def compute_pairwise_distances(landmarks: np.ndarray) -> np.ndarray:
    """Convert landmarks to pairwise distance representation (210-D per frame).

    Args:
        landmarks: Array of shape ``(T, 21, 3)``.

    Returns:
        Array of shape ``(T, 210)`` containing upper-triangular pairwise distances.
    """
    T = landmarks.shape[0]
    idx_i, idx_j = np.triu_indices(21, k=1)  # 21*20/2 = 210 pairs
    out = np.zeros((T, len(idx_i)), dtype=np.float32)
    for t in range(T):
        diffs = landmarks[t][idx_i] - landmarks[t][idx_j]  # (210, 3)
        out[t] = np.linalg.norm(diffs, axis=-1)
    return out


def preprocess_dataset(
    video_dir: str,
    output_dir: str,
    max_frames: int = 32,
    normalize_translation: bool = True,
    normalize_scale: bool = True,
) -> None:
    """Preprocess an entire dataset directory of videos.

    Expected structure::

        video_dir/
            class_0/
                video1.mp4
                video2.mp4
            class_1/
                ...

    Produces::

        output_dir/
            class_0/
                video1.npy
                ...

    Args:
        video_dir: Root directory containing class sub-folders with videos.
        output_dir: Root directory for saving ``.npy`` landmark files.
        max_frames: Fixed sequence length.
        normalize_translation: Wrist-centre each frame.
        normalize_scale: Scale-normalise each frame.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    classes = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes in {video_dir}")

    for cls_dir in tqdm(classes, desc="Classes"):
        cls_name = cls_dir.name
        out_cls = output_dir / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        videos = [f for f in cls_dir.iterdir() if f.suffix.lower() in video_exts]
        for vid_path in tqdm(videos, desc=f"  {cls_name}", leave=False):
            out_path = out_cls / (vid_path.stem + ".npy")
            if out_path.exists():
                continue
            lm = extract_landmarks_from_video(
                str(vid_path), max_frames=max_frames, hands_model=hands_model,
            )
            if lm is None:
                continue
            lm = normalize_landmarks(
                lm, translate=normalize_translation, scale=normalize_scale,
            )
            np.save(str(out_path), lm)

    hands_model.close()
    print(f"Preprocessing complete. Saved to {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess sign language videos")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Root dir with class sub-folders of videos")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npy landmark files")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--no_translate", action="store_true")
    parser.add_argument("--no_scale", action="store_true")
    args = parser.parse_args()

    preprocess_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        normalize_translation=not args.no_translate,
        normalize_scale=not args.no_scale,
    )
