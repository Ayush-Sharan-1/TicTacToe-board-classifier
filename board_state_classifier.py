import cv2
import numpy as np

from utils import visualize_board_detection

def load_image(path):
    """
    Load the input image from file.

    Args:
        path (str): Path to the image file.

    Returns:
        image (np.ndarray): Image in BGR format (as used by OpenCV).
    """
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image from path: {path}")

    # Optional: resize for consistent processing
    max_width = 640
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    return image

def detect_board(image):
    """
    Detect the Tic Tac Toe board region in the image.
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur + edge detection
    3. Find contours
    4. Select the largest 4-sided contour (the board)
    5. Extract corner points
    Returns:
        cropped board image (still perspective-skewed),
        list of 4 corner points (for rectification).
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Blur + edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 3. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found. Board not detected.")

    # 4. Pick the largest contour by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_contour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:  # look for quadrilateral
            board_contour = approx
            break

    if board_contour is None:
        raise ValueError("Board contour not found.")

    # 5. Extract corner points (flattened to (x, y))
    corners = board_contour.reshape(4, 2)

    # Optional: crop rough bounding box for debugging (not rectified yet)
    x, y, w, h = cv2.boundingRect(board_contour)
    cropped = image[y:y+h, x:x+w]

    return cropped, corners


def rectify_board(board_image, corners):
    """
    Apply homography (perspective transform) to warp the board
    into a top-down (rectified) square view.
    Returns: rectified board image.
    """
    pass

def split_cells(rectified_board):
    """
    Split the rectified board into a 3x3 grid of cell images.
    Returns: list of 9 images (cells) in row-major order.
    """
    pass

def classify_cell(cell_image):
    """
    Classify a single cell as 'X', 'O', or '_'.
    Uses color/shape features or a classifier.
    Returns: one of 'X', 'O', '_'.
    """
    pass

def get_board_state(cell_images):
    """
    Run classification for each cell.
    Returns: 3x3 numpy array or list of lists with 'X', 'O', '_'.
    """
    pass

def main(image_path):
    """
    High-level pipeline:
    1. Load image
    2. Detect board
    3. Rectify board
    4. Split into cells
    5. Classify each cell
    6. Print/return board state
    """
    image_path = "Images/B2_Br3.png"
    image = load_image(image_path)

    # Display the image
    # cv2.imshow("Loaded Image", image)
    # cv2.waitKey(0)   # Wait for a key press
    # cv2.destroyAllWindows()

    board, corners = detect_board(image)

    print("Detected corner points:\n", corners)
    # Visualize result
    visualize_board_detection(image, corners)

    # rectified = rectify_board(board, corners)
    # cells = split_cells(rectified)
    # state = get_board_state(cells)

    # print("Board state:")
    # for row in state:
    #     print(row)

if __name__ == "__main__":
    # Example usage
    main("tic_tac_toe.jpg")
