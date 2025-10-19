import cv2
import numpy as np
from utils import visualize_board_detection

def save_reference_features(ref_image_path, output_dir="data"):
    image = cv2.imread(ref_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from '{ref_image_path}'")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Detect ORB keypoints
    orb = cv2.ORB_create(2000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 2. Let the user select 4 corners interactively
    corners = select_corners(image)

    if corners is None or len(corners) != 4:
        print("Corner selection failed or cancelled. Exiting without saving reference data.")
        raise RuntimeError("Corner selection aborted by user")

    # 3. Save reference data
    np.save(f"{output_dir}/ref_keypoints.npy", np.array([kp.pt for kp in keypoints]))
    np.save(f"{output_dir}/ref_descriptors.npy", descriptors)
    np.save(f"{output_dir}/ref_corners.npy", corners)

    annotated = visualize_board_detection(image, corners)
    cv2.imwrite(f"{output_dir}/reference_board_marked.jpg", annotated)

    print("Reference features and corners saved successfully!")

def select_corners(image):
    """
    Display an image and let the user click 4 points in order:
    top-left, top-right, bottom-right, bottom-left.
    Press 'r' to reset or 'q' to quit without saving.
    """

    clone = image.copy()
    selected_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(clone, f"{len(selected_points)}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Select 4 Corners", clone)

    cv2.imshow("Select 4 Corners", clone)
    cv2.setMouseCallback("Select 4 Corners", click_event)

    print("Click the 4 board corners in order: TL → TR → BR → BL")
    print("Press 'r' to reset, 'q' to quit, or 'Enter' when done.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clone = image.copy()
            selected_points = []
            cv2.imshow("Select 4 Corners", clone)
            print("Reset points. Click again.")
        elif key == 13:  # Enter key
            if len(selected_points) == 4:
                break
            else:
                print(f"You selected {len(selected_points)} points. Need 4.")
        elif key == ord("q"):
            selected_points = []
            break

    cv2.destroyAllWindows()

    if len(selected_points) == 4:
        corners = np.array(selected_points, dtype=np.float32)
        print("Selected corners:", corners)
        return corners
    else:
        print("No valid corners selected.")
        return None


if __name__ == "__main__":
    save_reference_features("Images/Empty.png")
