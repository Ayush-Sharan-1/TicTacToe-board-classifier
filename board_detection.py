import cv2
import numpy as np

def load_reference_features(path="data"):
    """Load saved ORB reference data."""
    keypoints = np.load(f"{path}/ref_keypoints.npy")
    descriptors = np.load(f"{path}/ref_descriptors.npy")
    corners = np.load(f"{path}/ref_corners.npy")
    return keypoints, descriptors, corners

def load_reference_image(path="data"):
    """Load saved ORB reference image."""
    ref_image = cv2.imread(f"{path}/reference_board_marked.jpg")
    return ref_image

def estimate_homography(ref_kps, ref_desc, img_gray, ref_corners, debug=False):
    """
    Estimate the homography (perspective transform) between
    the reference image and the current grayscale image.

    Args:
        ref_kps (np.ndarray): Reference keypoint coordinates (N×2).
        ref_desc (np.ndarray): Reference descriptors (N×32).
        img_gray (np.ndarray): Current grayscale frame.
        ref_corners (np.ndarray): Reference corner coordinates (4×2).
        debug (bool): If True, visualize matches.

    Returns:
        H (np.ndarray): 3×3 homography matrix.
        detected_corners (np.ndarray): 4×2 array of detected board corners in current frame.
    """
    # Step 1. Detect ORB features in the current frame
    orb = cv2.ORB_create(2000)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    if descriptors is None or len(keypoints) == 0:
        print("No ORB features detected in current frame.")
        return None, None

    # Step 2. Match descriptors using brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_desc, descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print("Not enough matches found to estimate homography.")
        return None, None

    # Step 3. Extract matched points
    src_pts = np.float32([ref_kps[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Step 4. Compute homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography estimation failed.")
        return None, None

    # Step 5. Transform reference corners to current frame
    detected_corners = cv2.perspectiveTransform(ref_corners[None, :, :], H)[0]

    if debug:
        ref_image = load_reference_image()
        ref_keypoints_cv = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for x, y in ref_kps]
        vis = cv2.drawMatches(
            ref_image,
            ref_keypoints_cv,
            cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
            keypoints,
            matches[:50], None, flags=2
        )
        cv2.imshow("ORB Matches (debug)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return H, detected_corners

def detect_board_orb(image, debug=False):
    """
    Detect board in the given BGR image and return its corner coordinates.

    Args:
        image (np.ndarray): Input BGR frame.
        ref_kps (np.ndarray): Reference keypoint coordinates.
        ref_desc (np.ndarray): Reference ORB descriptors.
        ref_corners (np.ndarray): Reference board corner coordinates.
        debug (bool): Whether to visualize results.

    Returns:
        detected_corners (np.ndarray): 4×2 array of detected corner points.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load features saved features from reference image
    ref_kps, ref_desc, ref_corners = load_reference_features()

    # Estimate homography and get board corners
    H, detected_corners = estimate_homography(ref_kps, ref_desc, gray, ref_corners, debug=debug)

    if detected_corners is None:
        print("Board detection failed.")
        return None

    if debug:
        print("Detected corner points:\n", detected_corners)
        img_debug = image.copy()
        for (x, y) in detected_corners:
            cv2.circle(img_debug, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.polylines(img_debug, [np.int32(detected_corners)], True, (0, 255, 0), 2)
        cv2.imshow("Detected Board Corners", img_debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_corners