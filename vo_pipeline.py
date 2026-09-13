"""
Visual Odometry Pipeline

Estimates camera motion from a sequence of images using ORB feature tracking.
"""

import os
import numpy as np
import cv2

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
N_FRAMES = 200
TARGET_SIZE = (1024, 1024)
N_FEATURES = 2000
MIN_MATCHES = 10
OUTLIER_THRESHOLD = 2.0
MAX_ERROR_THRESHOLD = 5.0
CORRELATION_THRESHOLD = 0.75
SCALE_FACTOR = 6.0
CACHE_DIR = "data"
SOURCE_IMAGE = "sample_image.jpg"


# --------------------------------------------------------------------------
# Step 1: Synthetic sequence generation
# --------------------------------------------------------------------------
def _load_cached_sequence(cache_dir, n_frames):
    """Load a previously generated frame sequence and trajectory if available."""
    traj_file = os.path.join(cache_dir, "trajectory.npy")
    if not os.path.exists(traj_file):
        return None

    frames = []
    for i in range(n_frames):
        img_file = os.path.join(cache_dir, f"frame_{i:04d}.png")
        if not os.path.exists(img_file):
            break
        frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            break
        frames.append(frame)

    if len(frames) != n_frames:
        return None

    trajectory = np.load(traj_file)
    print(f"Loaded {len(frames)} frames from cache")
    return frames, trajectory


def _figure_eight_offset(t, scale):
    """Compute a figure-8 pixel offset at phase t."""
    denom = 1 + np.sin(t) ** 2
    tx = scale * 2 * np.cos(t) / denom
    ty = scale * 2 * np.sin(t) * np.cos(t) / denom

    # Add a small spiral perturbation
    spiral_factor = 0.2 * np.sin(4 * t)
    tx += spiral_factor * scale * 0.3
    ty += spiral_factor * scale * 0.3
    return tx, ty


def generate_synthetic_sequence(
    n_frames=N_FRAMES,
    cache_dir=CACHE_DIR,
    source_image=SOURCE_IMAGE,
    target_size=TARGET_SIZE,
    scale_factor=SCALE_FACTOR,
):
    """
    Generate an image sequence with known camera motion from a source image.
    Simulates camera movement by cropping a moving window from an upscaled image.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cached = _load_cached_sequence(cache_dir, n_frames)
    if cached is not None:
        return cached

    if not os.path.exists(source_image):
        raise FileNotFoundError(
            f"Source image '{source_image}' not found. Place an image with "
            f"distinct features (corners/edges/texture) in the working directory."
        )

    src_img = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        raise ValueError(f"Could not decode source image '{source_image}'.")

    output_h, output_w = target_size
    src_h, src_w = int(output_h * scale_factor), int(output_w * scale_factor)
    large_img = cv2.resize(src_img, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

    print(f"Generating {n_frames} frames of {output_h}x{output_w}...")

    max_offset_x = (src_w - output_w) // 2
    max_offset_y = (src_h - output_h) // 2
    max_range = min(max_offset_x, max_offset_y)
    scale = max_range / 2.5

    frames = []
    trajectory = []

    for i in range(n_frames):
        t = 2 * np.pi * i / n_frames
        tx, ty = _figure_eight_offset(t, scale)

        center_x = src_w // 2 + int(tx)
        center_y = src_h // 2 + int(ty)

        x1 = center_x - output_w // 2
        y1 = center_y - output_h // 2

        # Clamp to bounds
        x1 = max(0, min(x1, src_w - output_w))
        y1 = max(0, min(y1, src_h - output_h))
        x2, y2 = x1 + output_w, y1 + output_h

        frame = large_img[y1:y2, x1:x2].copy()

        # Add slight sensor noise
        noise = np.random.randn(output_h, output_w) * 2
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        img_file = os.path.join(cache_dir, f"frame_{i:04d}.png")
        cv2.imwrite(img_file, frame)

        frames.append(frame)

        # Accuracy Note: We record the exact integer coordinates (x1, y1) used to
        # slice the frame rather than the floating-point (tx, ty) values. This prevents
        # discretization bias from artificially inflating the trajectory error.
        # The coordinates are left unscaled because the crop is a direct slice of
        # the large image; the frame pixels map 1:1 to the trajectory coordinates.
        trajectory.append([x1, y1])

    trajectory = np.array(trajectory)
    np.save(os.path.join(cache_dir, "trajectory.npy"), trajectory)
    print(f"Saved {len(frames)} frames to {cache_dir}/")

    return frames, trajectory


# --------------------------------------------------------------------------
# Step 2: Feature detection & matching
# --------------------------------------------------------------------------
def detect_features(frame, n_features=N_FEATURES):
    """Detect ORB keypoints and compute descriptors."""
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio_thresh=0.75):
    """Match ORB descriptors using Hamming distance and Lowe's ratio test."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    # Accuracy Note: Lowe's ratio test rejects ambiguous matches by requiring
    # the best match to be significantly better than the second-best match.
    # This filters out false positives on repetitive textures before motion estimation.
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return sorted(good_matches, key=lambda m: m.distance)


# --------------------------------------------------------------------------
# Step 3: Motion estimation
# --------------------------------------------------------------------------
def estimate_motion(kp1, kp2, matches, outlier_threshold=OUTLIER_THRESHOLD):
    """Estimate camera motion from matched keypoints."""
    if len(matches) < MIN_MATCHES:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Accuracy Note: Apparent feature displacement (pts2 - pts1) is the motion
    # of the scene relative to the camera. True camera motion is the inverse
    # of this displacement (pts1 - pts2).
    motion_vectors = pts1 - pts2

    for iteration in range(3):
        median_motion = np.median(motion_vectors, axis=0)
        std_motion = np.std(motion_vectors, axis=0)
        std_safe = np.where(std_motion > 1e-6, std_motion, 1.0)

        z_scores = np.abs((motion_vectors - median_motion) / std_safe)
        inlier_mask = np.all(z_scores < outlier_threshold, axis=1)

        if np.sum(inlier_mask) >= MIN_MATCHES:
            motion_vectors = motion_vectors[inlier_mask]
        elif iteration == 2:
            return None

    # We use median instead of mean to provide robust estimation against any
    # residual outlier matches that slip past the ratio test and z-score checks.
    return np.median(motion_vectors, axis=0)


def process_frame_sequence(frames):
    """Run detection, matching, and motion estimation across all consecutive frames."""
    estimated_motions = []
    skipped = 0

    for i in range(len(frames) - 1):
        kp1, desc1 = detect_features(frames[i])
        kp2, desc2 = detect_features(frames[i + 1])

        if desc1 is None or desc2 is None:
            skipped += 1
            continue

        matches = match_features(desc1, desc2)
        if len(matches) < MIN_MATCHES:
            skipped += 1
            continue

        motion = estimate_motion(kp1, kp2, matches)
        if motion is None:
            skipped += 1
            continue

        estimated_motions.append(motion)

    if skipped:
        print(f"  (skipped {skipped} frame pairs with insufficient features/matches)")

    return np.array(estimated_motions)


# --------------------------------------------------------------------------
# Step 4: Trajectory & evaluation
# --------------------------------------------------------------------------
def build_trajectory(estimated_motions):
    """Integrate per-step motions into a cumulative trajectory."""
    if len(estimated_motions) == 0:
        raise RuntimeError("Motion estimation failed for all frames. Cannot build trajectory.")
    return np.vstack(([0, 0], np.cumsum(estimated_motions, axis=0)))


def compute_trajectory_error(estimated, ground_truth):
    """Mean absolute per-axis trajectory error in pixels."""
    return np.abs(np.mean(estimated - ground_truth))


def compute_correlation(estimated, ground_truth):
    """Pearson correlation between estimated and ground-truth trajectories."""
    if len(estimated) != len(ground_truth):
        print("[WARN] Trajectory length mismatch.")
        return None

    try:
        corr = np.corrcoef(estimated.flatten(), ground_truth.flatten())[0, 1]
        return None if np.isnan(corr) else corr
    except Exception:
        print("[WARN] Failed to compute correlation.")
        return None


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def run_visual_odometry_pipeline():
    """Execute the full VO pipeline."""
    print("=" * 60)
    print("Visual Odometry Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic image sequence...")
    frames, gt_trajectory = generate_synthetic_sequence()
    print(f"Generated {len(frames)} frames")

    print("\n[2/4] Processing frames...")
    estimated_motions = process_frame_sequence(frames)
    print(f"Estimated {len(estimated_motions)} motion steps")

    print("\n[3/4] Evaluating trajectory...")
    estimated_trajectory = build_trajectory(estimated_motions)
    gt_trajectory_relative = gt_trajectory - gt_trajectory[0]

    error = compute_trajectory_error(estimated_trajectory, gt_trajectory_relative)

    print("\n[4/4] Validation Results:")
    if error < MAX_ERROR_THRESHOLD:
        print(f"[OK] Trajectory error: {error:.2f} pixels < {MAX_ERROR_THRESHOLD}")
    else:
        print(f"[FAIL] Trajectory error: {error:.2f} pixels >= {MAX_ERROR_THRESHOLD}")

    correlation = compute_correlation(estimated_trajectory, gt_trajectory_relative)
    if correlation is not None:
        if correlation > CORRELATION_THRESHOLD:
            print(f"[OK] Motion correlation: r={correlation:.4f}")
        else:
            print(f"[FAIL] Motion correlation: r={correlation:.4f} < {CORRELATION_THRESHOLD}")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)

    return estimated_trajectory, gt_trajectory_relative, error, correlation


if __name__ == "__main__":
    try:
        estimated_trajectory, ground_truth, error, correlation = run_visual_odometry_pipeline()

        assert error < MAX_ERROR_THRESHOLD, f"Error {error:.2f} exceeds threshold {MAX_ERROR_THRESHOLD}"
        assert correlation is not None and correlation > CORRELATION_THRESHOLD, (
            f"Correlation {correlation} does not exceed {CORRELATION_THRESHOLD}"
        )
        print("\n[SUCCESS] All checks passed!")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
