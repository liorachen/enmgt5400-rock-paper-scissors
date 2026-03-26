# Preprocessing Script for RPS Dataset
# ENMGT 5400 - Rock Paper Scissors Project
# Written by: Claude (Anthropic) for Nish's project
#
# Step 1: Manual review — view each image, press K to keep or D to delete
# Step 2: Downsample kept images to 32x32 using the PDF method:
#           center-crop 160x120 → 96x96, then take every 3rd pixel → 32x32
# Step 3: Save to dataset/{rock,paper,scissors}/ as BMP
# Step 4: Print validation summary
#
# Requirements: pip install opencv-python numpy

import cv2
import numpy as np
import os
import shutil

# ---- CONFIGURATION ----
CAPTURES_DIR = 'captures'   # input: raw 160x120 BMPs
DATASET_DIR  = 'dataset'    # output: 32x32 BMPs after review
CLASSES      = ['rock', 'paper', 'scissors']
DISPLAY_SCALE = 6           # show images at 192x192 for easy viewing (32*6)
# ------------------------

def center_crop_96(img):
    """Crop 160x120 image to 96x96 by removing equal borders."""
    h, w = img.shape[:2]   # 120, 160
    top  = (h - 96) // 2   # 12
    left = (w - 96) // 2   # 32
    return img[top:top+96, left:left+96]

def every_third_pixel(img):
    """Downsample 96x96 to 32x32 by taking every 3rd pixel (PDF method)."""
    return img[::3, ::3]

def preprocess(img):
    """Full pipeline: 160x120 → center crop 96x96 → every 3rd pixel → 32x32."""
    cropped    = center_crop_96(img)
    downsampled = every_third_pixel(cropped)
    return downsampled

def review_phase():
    """
    Show each raw image. Press:
      K = keep
      D = delete (moves to captures/deleted/)
      Q = quit review early
    Returns count of kept/deleted per class.
    """
    print("\n=== PHASE 1: MANUAL REVIEW ===")
    print("Controls: K = keep | D = delete | Q = quit early")
    print("Review each image and remove blurry, misframed, or ambiguous ones.\n")

    deleted_dir = os.path.join(CAPTURES_DIR, 'deleted')
    os.makedirs(deleted_dir, exist_ok=True)

    stats = {}
    for cls in CLASSES:
        cls_dir = os.path.join(CAPTURES_DIR, cls)
        files   = sorted([f for f in os.listdir(cls_dir) if f.endswith('.bmp')])
        kept = deleted = 0

        print("--- Class: {} ({} images) ---".format(cls, len(files)))
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            img   = cv2.imread(fpath)
            if img is None:
                print("  WARNING: could not read", fpath)
                continue

            # Show at 6x scale for easy inspection
            display = cv2.resize(img, (img.shape[1] * DISPLAY_SCALE, img.shape[0] * DISPLAY_SCALE),
                                 interpolation=cv2.INTER_NEAREST)
            label = "{} | {} | K=keep D=delete Q=quit".format(cls, fname)
            cv2.putText(display, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Review', display)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('k') or key == ord('K'):
                    kept += 1
                    break
                elif key == ord('d') or key == ord('D'):
                    dst = os.path.join(deleted_dir, '{}_{}'.format(cls, fname))
                    shutil.move(fpath, dst)
                    deleted += 1
                    break
                elif key == ord('q') or key == ord('Q'):
                    print("  Quit early — {} kept, {} deleted so far".format(kept, deleted))
                    cv2.destroyAllWindows()
                    stats[cls] = (kept, deleted)
                    return stats

        print("  {} kept, {} deleted".format(kept, deleted))
        stats[cls] = (kept, deleted)

    cv2.destroyAllWindows()
    return stats

def downsample_phase():
    """
    Convert all kept raw images to 32x32 and save to dataset/.
    Uses PDF method: center-crop to 96x96, then take every 3rd pixel.
    """
    print("\n=== PHASE 2: DOWNSAMPLE TO 32x32 ===")

    for cls in CLASSES:
        src_dir = os.path.join(CAPTURES_DIR, cls)
        dst_dir = os.path.join(DATASET_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(src_dir) if f.endswith('.bmp')])
        count = 0
        for fname in files:
            fpath = os.path.join(src_dir, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue

            small = preprocess(img)

            # Verify output size
            assert small.shape == (32, 32, 3), \
                "Expected 32x32x3, got {} for {}".format(small.shape, fpath)

            out_path = os.path.join(dst_dir, fname)
            cv2.imwrite(out_path, small)
            count += 1

        print("  {}: {} images → dataset/{}/".format(cls, count, cls))

def validation_summary():
    """Print final counts and flag any issues."""
    print("\n=== PHASE 3: VALIDATION SUMMARY ===")

    total = 0
    issues = []
    for cls in CLASSES:
        dst_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(dst_dir):
            issues.append("MISSING: dataset/{}/".format(cls))
            continue

        files = [f for f in os.listdir(dst_dir) if f.endswith('.bmp')]
        count = len(files)
        total += count

        # Spot-check a few images for correct size
        for fname in files[:5]:
            img = cv2.imread(os.path.join(dst_dir, fname))
            if img is None:
                issues.append("UNREADABLE: dataset/{}/{}".format(cls, fname))
            elif img.shape != (32, 32, 3):
                issues.append("WRONG SIZE {}: dataset/{}/{}".format(img.shape, cls, fname))

        status = "OK" if count >= 100 else "LOW (want 100+)"
        print("  {:10s}: {:3d} images  [{}]".format(cls, count, status))

    print("  {:10s}: {:3d} images total".format("TOTAL", total))

    if total >= 300:
        print("\n  Rubric check: 300+ images PASSED ({})".format(total))
    else:
        print("\n  Rubric check: 300+ images FAILED — only {} (collect more)".format(total))

    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print("    -", issue)
    else:
        print("  No issues found. Dataset ready for training.")

if __name__ == '__main__':
    import sys
    skip_review = '--no-review' in sys.argv

    if skip_review:
        print("Skipping review phase (--no-review flag set).")
    else:
        review_stats = review_phase()
        print("\nReview complete:")
        for cls, (kept, deleted) in review_stats.items():
            print("  {:10s}: {} kept, {} deleted".format(cls, kept, deleted))

    downsample_phase()
    validation_summary()

    print("\nDataset saved to: {}/".format(DATASET_DIR))
    print("Next step: run train.py to train the CNN.")
