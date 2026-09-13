# Intern Task: Visual Odometry Debug & Refactor

## Context

You're working on SLAM for a drone in GPS-denied environments. Visual odometry (estimating camera motion from images) is a foundational SLAM technique.

## Task

The original repository contained a visual odometry implementation with bugs and limited error handling. The implementation has been debugged and refactored to:

1. Fix the trajectory and camera-motion bugs.
2. Refactor the pipeline into clean, modular functions.
3. Add basic error handling and logging.
4. Prevent individual frame-processing failures from breaking the complete trajectory.
5. Validate the estimated trajectory against the ground truth using meaningful metrics.

## Prerequisites

Before running, ensure you have `sample_image.jpg` in this folder. The image should contain distinct features such as corners, edges, or textures. A photo of a room, outdoor scene, or object-rich environment works well.

The pipeline upscales the source image by 6x and extracts 1024x1024 moving windows to simulate camera motion without artificial borders or black boxes.

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
python vo_pipeline.py
```

The first run generates 200 synthetic frames from `sample_image.jpg` and stores them in the `data/` directory. The generated ground-truth trajectory is also cached there.

Subsequent runs reuse the cached frames for faster execution.

To regenerate the synthetic sequence:

```bash
rm -rf data
```

On Windows, delete the `data/` folder manually.

## Implementation Overview

The refactored pipeline is divided into the following stages:

### 1. Synthetic Sequence Generation

`generate_synthetic_sequence()`:

- Creates the cache directory safely.
- Loads cached frames when available.
- Detects missing or corrupted cached images.
- Validates that the source image exists and can be decoded.
- Upscales the source image.
- Generates a moving 1024x1024 window following the predefined camera-motion path.
- Adds small random image noise.
- Saves generated frames and the ground-truth trajectory.

### 2. Feature Extraction and Matching

`extract_and_match()`:

- Validates input frames.
- Detects ORB features.
- Computes ORB descriptors.
- Matches descriptors using a Hamming-distance BFMatcher.
- Handles missing descriptors and OpenCV failures safely.

### 3. Camera Motion Estimation

`estimate_camera_motion()`:

- Rejects frame pairs with too few matches.
- Converts matched keypoints into point coordinates.
- Estimates the dominant translation from the matched feature motion.
- Uses iterative median/std-based outlier rejection.
- Falls back to the median motion when the inlier count becomes too small instead of dropping the frame.

The motion vector is calculated as:

```text
motion = points_frame_1 - points_frame_2
```

This keeps the estimated translation consistent with the camera-frame convention used by the generated ground truth.

### 4. Sequence Processing

`process_sequence()`:

- Processes every consecutive frame pair.
- Keeps the number of estimated motion steps consistent with the input sequence.
- Uses the previous valid motion, or zero motion for the first failure, when a frame pair cannot be processed.
- Prevents isolated processing errors from shortening the estimated trajectory.

### 5. Trajectory Evaluation

`compute_trajectory_error()` now calculates the mean Euclidean distance between the estimated and ground-truth trajectory points:

```text
distance = ||estimated - ground_truth||
error = mean(distance)
```

This replaces the original calculation based on the absolute mean of coordinate differences, which could hide spatial errors through cancellation between positive and negative values.

`evaluate_metrics()` also computes the correlation between the estimated and ground-truth trajectories and guards against zero-variance inputs.

## Important Bugs Fixed

### 1. Incorrect Ground-Truth Scaling

The original implementation divided the stored ground-truth translation by `scale_factor`. The generated frame coordinates were already expressed in the same image-coordinate system used for the estimated feature motion.

**Fix:** Store `[tx, ty]` directly as the ground-truth trajectory.

### 2. Incorrect Motion Direction

The original implementation used:

```python
motion_vectors = pts2 - pts1
```

This produced motion in the opposite direction to the camera-frame convention used by the ground truth.

**Fix:**

```python
motion_vectors = pts1 - pts2
```

### 3. Trajectory Length Mismatch

The original processing loop used `continue` when a frame pair failed. This could produce fewer motion steps than expected and cause the estimated and ground-truth trajectories to have different lengths.

**Fix:** Failed frame pairs now use the previous valid motion, or `[0.0, 0.0]` if no previous motion exists.

### 4. Incorrect Trajectory Error Metric

The original implementation used:

```python
np.mean(estimated - ground_truth)
```

followed by `abs()`. This is not a proper spatial trajectory error because errors in different directions can cancel each other.

**Fix:** Calculate the Euclidean distance at every trajectory point and then take the mean.

### 5. Fragile Cache Loading

The original implementation assumed cached images could always be read successfully.

**Fix:** Cached images are checked for successful decoding. Missing or corrupted cache data causes the sequence to be regenerated.

### 6. Missing Source-Image Validation

The original implementation relied on OpenCV to fail if `sample_image.jpg` did not exist.

**Fix:** The source image is explicitly checked and a clear `FileNotFoundError` is raised.

### 7. Insufficient OpenCV Error Handling

Feature detection, descriptor matching, resizing, and frame writing can fail due to invalid input or OpenCV errors.

**Fix:** Relevant OpenCV operations are wrapped with targeted error handling and meaningful log messages.

### 8. Lack of Structured Logging

The original code relied heavily on `print()` statements.

**Fix:** Python's `logging` module is used for warnings, errors, and informational messages while keeping the main pipeline output readable.

### 9. Monolithic Pipeline Structure

The original implementation combined frame processing and metric evaluation logic inside the main orchestration function.

**Fix:** Responsibilities are separated into:

- `generate_synthetic_sequence()`
- `extract_and_match()`
- `estimate_camera_motion()`
- `process_sequence()`
- `compute_trajectory_error()`
- `evaluate_metrics()`
- `print_evaluation_report()`
- `run_visual_odometry_pipeline()`

The final `run_visual_odometry_pipeline()` function now primarily orchestrates these components.

## Validation

The pipeline reports:

- Number of generated frames.
- Number of estimated motion steps.
- Mean trajectory error in pixels.
- Motion correlation coefficient.
- Whether the trajectory error is below the configured threshold.
- Whether the motion correlation exceeds `0.75`.

The configured acceptance threshold is:

```text
Trajectory error < 5.0 pixels
Motion correlation > 0.75
```

A successful run ends with:

```text
[SUCCESS] All checks passed!
```

## Output and Error Handling

The program distinguishes between different failure cases:

- Missing input image → clear failure message.
- Invalid/corrupted cached data → warning followed by regeneration.
- Insufficient feature matches → fallback motion is used.
- Processing errors for an individual frame pair → previous/zero motion is used.
- Threshold failure → assertion failure and non-zero exit status.
- Keyboard interruption → cleanly reports that the pipeline was aborted.
- Unexpected fatal errors → logged with a traceback.


