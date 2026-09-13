import numpy as np
import cv2
import os
# ADDED: logging module for production-ready, severity-based output[cite: 1].
# REMOVED: np.random.seed(42) was removed to ensure synthetic noise generation is genuinely random on each run[cite: 1, 2].
import logging

# Configuration
N_FRAMES = 200
TARGET_SIZE = (1024, 1024)
N_FEATURES = 2000
MIN_MATCHES = 10
OUTLIER_THRESHOLD = 2.0
MAX_ERROR_THRESHOLD = 5.0

# ADDED: Setup basic logging configuration to replace standard print() statements, making debugging easier[cite: 1].
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def generate_synthetic_sequence(n_frames=N_FRAMES, cache_dir="data", source_image="sample_image.jpg"):
    """Generates synthetic image sequence with known camera motion."""
    # CHANGED: Wrapped directory creation in a try-except block to prevent silent or cryptical crashes on filesystem permission issues[cite: 1, 2].
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create cache directory '{cache_dir}': {e}")

    traj_file = os.path.join(cache_dir, "trajectory.npy")

    # Try loading from cache
    # CHANGED: Added extensive try-except block around cache loading to handle empty/corrupted cache files gracefully without crashing[cite: 1, 2].
    try:
        if os.path.exists(traj_file):
            frames = []
            for i in range(n_frames):
                img_file = os.path.join(cache_dir, f"frame_{i:04d}.png")
                if not os.path.exists(img_file):
                    break
                frame = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

                # CHANGED: Added check for None to warn about corrupted images instead of blindly appending them[cite: 1, 2].
                if frame is not None:
                    frames.append(frame)
                else:
                    logging.warning(f"Corrupted image in cache: {img_file}")
                    break

            if len(frames) == n_frames:
                trajectory = np.load(traj_file)
                logging.info(f"Loaded {len(frames)} frames from cache")
                return frames, trajectory
    except Exception as e:
        logging.warning(f"Failed to load cache: {e}. Regenerating sequence...")

    # ---------------------------------------------------------
    # Error Handling if image does not exist.
    # ---------------------------------------------------------
    if not os.path.exists(source_image):
        raise FileNotFoundError(
            f"Source image '{source_image}' not found. Place an image with "
            f"distinct features (corners/edges/texture) in the working directory."
        )

    src_img = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        raise ValueError(f"Could not decode source image '{source_image}'.")
    # ---------------------------------------------------------
    output_h, output_w = TARGET_SIZE
    scale_factor = 6.0
    src_h, src_w = int(output_h * scale_factor), int(output_w * scale_factor)

    # CHANGED: Wrapped OpenCV resize in a try-except to catch OpenCV faults[cite: 1].
    try:
        large_img = cv2.resize(src_img, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        raise RuntimeError(f"Failed to resize source image: {e}")

    logging.info(f"Generating {n_frames} frames of {output_h}x{output_w}...")

    frames = []
    trajectory = []

    # CHANGED: Simplified max_range calculation for scaling the spiral movement[cite: 1, 2].
    max_range = min((src_w - output_w) // 2, (src_h - output_h) // 2)
    scale = max_range / 2.5

    for i in range(n_frames):
        t = 2 * np.pi * i / n_frames
        denom = 1 + np.sin(t) ** 2
        tx = scale * 2 * np.cos(t) / denom
        ty = scale * 2 * np.sin(t) * np.cos(t) / denom

        spiral_factor = 0.2 * np.sin(4 * t)
        tx += spiral_factor * scale * 0.3
        ty += spiral_factor * scale * 0.3

        center_x = src_w // 2 + int(tx)
        center_y = src_h // 2 + int(ty)

        # CHANGED: Consolidated redundant boundary clamping logic into cleaner max/min operations for x1 and y1[cite: 1, 2].
        x1 = max(0, min(center_x - output_w // 2, src_w - output_w))
        y1 = max(0, min(center_y - output_h // 2, src_h - output_h))

        frame = large_img[y1:y1+output_h, x1:x1+output_w].copy()
        noise = np.random.randn(output_h, output_w) * 2
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        # Handle disk write errors
        # CHANGED: Wrapped cv2.imwrite in try-except so disk I/O faults do not crash the pipeline[cite: 1].
        try:
            cv2.imwrite(os.path.join(cache_dir, f"frame_{i:04d}.png"), frame)
        except cv2.error as e:
            logging.error(f"Failed to write frame {i} to disk: {e}")

        frames.append(frame)

        # CHANGED: Appending [tx, ty] directly instead of [tx / scale_factor, ty / scale_factor] to correct a mathematical bug in how ground truth coordinates were recorded[cite: 1, 2].
        trajectory.append([tx, ty])

    trajectory = np.array(trajectory)

    # CHANGED: Wrapped np.save in IOError check[cite: 1].
    try:
        np.save(traj_file, trajectory)
    except IOError as e:
        logging.error(f"Failed to save trajectory to disk: {e}")

    return frames, trajectory


# CHANGED: Combined the old `detect_features` and `match_features` into a single logical grouping[cite: 1, 2].
def extract_and_match(frame1, frame2):
    """Detects features and finds robust matches between two frames."""
    # ADDED: Input validation to ensure frames are provided[cite: 1].
    if frame1 is None or frame2 is None:
        raise ValueError("One or both input frames are None.")

    # CHANGED: Wrapped ORB creation and computation in try-except to prevent OpenCV crashes on empty arrays[cite: 1].
    try:
        orb = cv2.ORB_create(nfeatures=N_FEATURES)
        kp1, desc1 = orb.detectAndCompute(frame1, None)
        kp2, desc2 = orb.detectAndCompute(frame2, None)
    except cv2.error as e:
        logging.error(f"OpenCV error during feature detection: {e}")
        return [], [], []

    # ADDED: Fallback returns if descriptors are missing or empty[cite: 1].
    if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
        return kp1, kp2, []

    # CHANGED: Wrapped BFMatcher in try-except to prevent OpenCV crashes[cite: 1].
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(desc1, desc2), key=lambda x: x.distance)
        return kp1, kp2, matches
    except cv2.error as e:
        logging.error(f"OpenCV error during matching: {e}")
        return kp1, kp2, []


# CHANGED: Renamed from estimate_motion to estimate_camera_motion[cite: 1, 2].
def estimate_camera_motion(kp1, kp2, matches):
    """Estimates camera translation vector from keypoint matches."""
    # ADDED: Early exit if insufficient matches are provided[cite: 1].
    if not matches or len(matches) < MIN_MATCHES:
        return None

    # CHANGED: Added try-except IndexError to protect against mismatched keypoints and match indices[cite: 1].
    try:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    except IndexError as e:
        logging.error(f"Mismatch between keypoints and match indices: {e}")
        return None

    # CHANGED: Reversed the vector calculation from 'pts2 - pts1' to 'pts1 - pts2' to mathematically correct the camera's frame-of-reference motion[cite: 1, 2].
    motion_vectors = pts1 - pts2

    for _ in range(3):
        median_motion = np.median(motion_vectors, axis=0)
        std_motion = np.std(motion_vectors, axis=0)

        # Prevent divide-by-zero if standard deviation is zero
        std_safe = np.where(std_motion > 1e-6, std_motion, 1.0)

        z_scores = np.abs((motion_vectors - median_motion) / std_safe)
        inlier_mask = np.all(z_scores < 1.5, axis=1)

        if np.sum(inlier_mask) >= MIN_MATCHES:
            motion_vectors = motion_vectors[inlier_mask]
        else:
            # CHANGED: Instead of returning None and dropping the frame entirely, it now falls back to returning the median_motion to maintain trajectory sequence length[cite: 1, 2].
            return median_motion

    return np.mean(motion_vectors, axis=0)


# CHANGED: Abstracted the main processing loop out of the orchestration function into this dedicated function[cite: 1, 2].
def process_sequence(frames):
    """Processes a sequence of frames and returns the estimated frame-to-frame motions."""
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("A minimum of 2 frames is required to process a sequence.")

    estimated_motions = []

    for i in range(len(frames) - 1):
        try:
            kp1, kp2, matches = extract_and_match(frames[i], frames[i + 1])
            motion = estimate_camera_motion(kp1, kp2, matches)

            # CHANGED: Fixed the critical "trajectory length mismatch" bug from the old script. Instead of using 'continue' to skip failed motions, it defaults to the previous motion (or zero)[cite: 1, 2].
            if motion is None:
                logging.warning(f"Insufficient matches between frame {i} and {i+1}. Using previous motion.")
                motion = estimated_motions[-1] if estimated_motions else np.array([0.0, 0.0])

            estimated_motions.append(motion)

        except Exception as e:
            logging.error(f"Critical error processing frames {i} to {i+1}: {e}")
            motion = estimated_motions[-1] if estimated_motions else np.array([0.0, 0.0])
            estimated_motions.append(motion)

    return np.array(estimated_motions)


def compute_trajectory_error(estimated, ground_truth):
    """Computes average Euclidean distance error between trajectories."""
    if estimated.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: estimated shape {estimated.shape} vs ground truth shape {ground_truth.shape}")

    # CHANGED: Replaced mathematically flawed np.mean(estimated - ground_truth) with actual spatial Euclidean distance tracking[cite: 1, 2].
    distances = np.linalg.norm(estimated - ground_truth, axis=1)
    return np.mean(distances)


# CHANGED: Extracted error and correlation calculations from the main function into this separate, clean function[cite: 1, 2].
def evaluate_metrics(estimated_trajectory, gt_trajectory_relative):
    """Calculates error and correlation between estimated and ground truth trajectories."""
    try:
        error = compute_trajectory_error(estimated_trajectory, gt_trajectory_relative)
    except ValueError as e:
        logging.error(f"Cannot compute trajectory error: {e}")
        return float('inf'), 0.0

    # ADDED: Guard to prevent division-by-zero or RuntimeWarning when calculating correlation on flat (zero standard deviation) trajectory data[cite: 1].
    if np.std(estimated_trajectory) == 0 or np.std(gt_trajectory_relative) == 0:
        logging.warning("Trajectory standard deviation is zero. Correlation undefined.")
        correlation = 0.0
    else:
        correlation = np.corrcoef(
            estimated_trajectory.flatten(),
            gt_trajectory_relative.flatten()
        )[0, 1]

    return error, correlation


# CHANGED: Moved all result printing statements here to strictly adhere to the Single Responsibility Principle[cite: 1, 2].
def print_evaluation_report(error, correlation):
    """Prints the final validation metrics."""
    print("\n[4/4] Validation Results:")
    if error < MAX_ERROR_THRESHOLD:
        print(f"[OK] Trajectory error: {error:.4f} pixels < {MAX_ERROR_THRESHOLD}")
    else:
        print(f"[FAIL] Trajectory error: {error:.4f} pixels >= {MAX_ERROR_THRESHOLD}")

    if correlation > 0.75:
        print(f"[OK] Motion correlation: r={correlation:.4f}")
    else:
        print(f"[FAIL] Motion correlation: r={correlation:.4f} < 0.75")


def run_visual_odometry_pipeline():
    """Main pipeline execution orchestrator."""
    # CHANGED: This function now acts strictly as an orchestrator, delegating logic to the newly extracted modular functions[cite: 1, 2].
    print("=" * 60)
    print("Visual Odometry Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic image sequence...")
    frames, gt_trajectory = generate_synthetic_sequence()

    if len(frames) < 2:
        raise RuntimeError("Failed to generate enough frames to run the pipeline.")

    print("\n[2/4] Processing frames...")
    estimated_motions = process_sequence(frames)
    print(f"Estimated {len(estimated_motions)} motion steps")

    print("\n[3/4] Evaluating trajectory...")
    estimated_trajectory = np.vstack(([0, 0], np.cumsum(estimated_motions, axis=0)))
    gt_trajectory_relative = gt_trajectory - gt_trajectory[0]

    error, correlation = evaluate_metrics(estimated_trajectory, gt_trajectory_relative)
    print_evaluation_report(error, correlation)

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)

    # CHANGED: Now only returns the error value instead of the full trajectories, simplifying the output signature[cite: 1, 2].
    return error


if __name__ == "__main__":
    # CHANGED: Replaced the broad, generic Exception catch with specific handling for clean user aborts (KeyboardInterrupt), threshold failures (AssertionError), and logging for fatals[cite: 1, 2].
    try:
        error = run_visual_odometry_pipeline()
        assert error < MAX_ERROR_THRESHOLD, f"Error {error:.4f} exceeds threshold."
        print("\n[SUCCESS] All checks passed!")
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion Error: {ae}")
        exit(1)
    except FileNotFoundError as fnfe:
        print(f"\n[FAIL] Missing Data: {fnfe}")
        exit(1)
    except KeyboardInterrupt:
        print("\n[ABORTED] Pipeline stopped by user.")
        exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline crashed: {e}")
        logging.exception("Detailed traceback:")
        exit(1)
