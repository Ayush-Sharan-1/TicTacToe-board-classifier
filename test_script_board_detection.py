import cv2
import numpy as np
import os
from pathlib import Path

from board_detection import detect_board_orb
from utils import visualize_board_detection  # assumed function to visualize detection


# ---------- CONFIG ----------
TEST_DIR = Path("Images")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Specific images to augment
AUGMENT_LIST = [
    "test_images/wcs.png",
]

# Random seed for reproducibility
np.random.seed(42)


# ---------- AUGMENTATION FUNCTIONS ----------
def add_noise(image, sigma=15):
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def change_brightness(image, factor=1.3):
    """Increase or decrease brightness by a factor."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def rotate_image(image, angle=15):
    """Rotate image by a given angle around its center."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def add_random_blobs(image, num_blobs=5):
    """Add random circular blobs (simulate occlusion)."""
    img = image.copy()
    h, w = img.shape[:2]
    for _ in range(num_blobs):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        radius = np.random.randint(10, 40)
        color = (np.random.randint(0, 255),
                 np.random.randint(0, 255),
                 np.random.randint(0, 255))
        cv2.circle(img, center, radius, color, -1)
    return img


# ---------- MAIN LOOP ----------
def main():
    # Load reference data (assuming saved in data/)
    from board_detection import load_reference_features
    ref_kps, ref_desc, ref_corners = load_reference_features("data")

    image_files = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Could not load: {img_path}")
            continue

        print(f"▶ Processing: {img_path.name}")

        # Run detection on original
        corners = detect_board_orb(image)
        result_img = visualize_board_detection(image, corners)
        cv2.imwrite(str(RESULTS_DIR / f"{img_path.stem}_orig.jpg"), result_img)

        # Apply augmentations if in AUGMENT_LIST
        if str(img_path) in AUGMENT_LIST:
            augmentations = {
                "noise": add_noise(image),
                "bright": change_brightness(image, 1.5),
                "dark": change_brightness(image, 0.6),
                "rot": rotate_image(image, 180),
                "blobs": add_random_blobs(image),
            }

            for name, aug_img in augmentations.items():
                corners = detect_board_orb(aug_img)
                result_img = visualize_board_detection(aug_img, corners)
                out_path = RESULTS_DIR / f"{img_path.stem}_{name}.jpg"
                cv2.imwrite(str(out_path), result_img)
                print(f"  ⤷ Saved augmented result: {out_path.name}")

    print("\n✅ All images processed. Check 'results/' folder.")


if __name__ == "__main__":
    main()
