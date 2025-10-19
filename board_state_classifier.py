import cv2
import numpy as np

from utils import visualize_board_detection
from board_detection import detect_board_orb

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

def rectify_board(image, corners, output_size=300, debug=False):
    """
    Rectifythe Tic Tac Toe board to a top-down view.

    Args:
        image (np.ndarray): Original BGR image.
        corners (np.ndarray): 4x2 array of board corner coordinates (TL, TR, BR, BL).
        output_size (int): Size of the output square board in pixels.
        debug (bool): If True, visualize warped result.

    Returns:
        rectified (np.ndarray): Warped board image of shape (output_size, output_size, 3).
        H (np.ndarray): Homography matrix used for warping.
    """

    if corners is None or len(corners) != 4:
        raise ValueError("warp_board() expects 4 corner points.")

    # Ensure float32 for OpenCV
    corners = np.array(corners, dtype=np.float32)

    # Order corners consistently: TL, TR, BR, BL
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    top_left = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right = corners[np.argmin(diff)]
    bottom_left = corners[np.argmax(diff)]
    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    # Define target points (perfect square)
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    # Compute homography and warp
    H = cv2.getPerspectiveTransform(ordered, dst_pts)
    rectified = cv2.warpPerspective(image, H, (output_size, output_size))

    if debug:
        cv2.imshow("Warped Board", rectified)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rectified, H


import numpy as np

def split_cells(rectified_board, debug=False):
    """
    Split the rectified (top-down) Tic Tac Toe board into 3x3 cell images.

    Args:
        rectified_board (np.ndarray): Warped board image (square).
        debug (bool): If True, display each cell in a 3x3 grid layout.

    Returns:
        cells (list): List of 9 cropped cell images in row-major order:
                      [row0col0, row0col1, row0col2,
                       row1col0, row1col1, row1col2,
                       row2col0, row2col1, row2col2]
    """
    if rectified_board is None:
        raise ValueError("split_cells() received None as input image.")

    h, w = rectified_board.shape[:2]
    cell_h = h // 3
    cell_w = w // 3

    cells = []
    for row in range(3):
        for col in range(3):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            cell = rectified_board[y1:y2, x1:x2].copy()
            cells.append(cell)

    # Debug visualization
    if debug:
        # Parameters
        gap = 10                # pixel gap between cells
        cell_size = 100         # display size of each cell
        grid_size = 3

        # Resize all cells for uniform display
        cell_resized = [cv2.resize(c, (cell_size, cell_size)) for c in cells]

        # Create vertical and horizontal gaps (white background)
        h_gap = 255 * np.ones((cell_size, gap, 3), dtype=np.uint8)
        v_gap = 255 * np.ones((gap, (cell_size * grid_size) + (gap * (grid_size - 1)), 3), dtype=np.uint8)

        # Build each row with horizontal gaps
        rows = []
        for i in range(0, 9, 3):
            row = np.hstack([
                cell_resized[i],
                h_gap,
                cell_resized[i + 1],
                h_gap,
                cell_resized[i + 2]
            ])
            rows.append(row)

        # Stack rows with vertical gaps
        grid_image = np.vstack([
            rows[0],
            v_gap,
            rows[1],
            v_gap,
            rows[2]
        ])

        cv2.imshow("Split Cells (3x3 Grid with Gaps)", grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cells



import cv2
import numpy as np

def classify_cell(cell_image, debug=False):
    """
    Classify a Tic Tac Toe cell as 'black', 'brown', or 'empty'
    based on dominant color region.

    Args:
        cell_image (np.ndarray): Cell image (BGR).
        debug (bool): If True, show intermediate visualization.

    Returns:
        str: One of {'black', 'brown', 'empty'}
    """
    if cell_image is None:
        return 'empty'

    # Convert to HSV for robust color detection
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)

    # Define color thresholds (tuned for typical lighting)
    # Black: low brightness
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])

    # Brown: hue near orange/red, moderate brightness
    lower_brown = np.array([5, 40, 120])
    upper_brown = np.array([30, 200, 230])

    # Create masks
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Compute relative coverage in the cell
    total_pixels = cell_image.shape[0] * cell_image.shape[1]
    black_ratio = np.sum(mask_black > 0) / total_pixels
    brown_ratio = np.sum(mask_brown > 0) / total_pixels

    # Decision thresholds
    if black_ratio > 0.30 and black_ratio > brown_ratio:
        label = 'black'
    elif brown_ratio > 0.10 and brown_ratio > black_ratio:
        label = 'brown'
    else:
        label = 'empty'

    if debug:
        combined = np.hstack([
            cell_image,
            cv2.cvtColor(mask_black, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_brown, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow("Cell / BlackMask / BrownMask", combined)
        print(f"Black: {black_ratio:.2f}, Brown: {brown_ratio:.2f}, Label: {label}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return label

def get_board_state(cell_images, brown_symbol='O', black_symbol='X', empty_symbol='_', debug=False):
    """
    Classify all 9 cell images into board state.

    Args:
        cell_images (list[np.ndarray]): List of 9 cell images in row-major order.
        brown_symbol (str): Symbol to assign to brown cells (default 'O').
        black_symbol (str): Symbol to assign to black cells (default 'X').
        empty_symbol (str): Symbol to assign to empty cells (default '_').
        debug (bool): If True, print classification per cell.

    Returns:
        board (np.ndarray): 3x3 array with symbols ('X', 'O', '_').
    """
    if len(cell_images) != 9:
        raise ValueError(f"Expected 9 cell images, got {len(cell_images)}")

    board = np.empty((3, 3), dtype='<U1')  # 3x3 array of single-character strings

    for i, cell in enumerate(cell_images):
        color_label = classify_cell(cell, debug=False)

        if color_label == 'brown':
            symbol = brown_symbol
        elif color_label == 'black':
            symbol = black_symbol
        else:
            symbol = empty_symbol

        row, col = divmod(i, 3)
        board[row, col] = symbol

        if debug:
            print(f"Cell {i} ({row},{col}) → {color_label} → {symbol}")

    return board


def main():
    """
    High-level pipeline:
    1. Load image
    2. Detect board
    3. Rectify board
    4. Split into cells
    5. Classify each cell
    6. Print/return board state
    """
    image_path = "Images/wcs.png"
    image = load_image(image_path)

    corners = detect_board_orb(image, debug=True)

    rectified, _ = rectify_board(image, corners, output_size=300, debug=False)

    cells = split_cells(rectified, debug=False)

    state = get_board_state(cells)

    print("Board state:")
    for row in state:
        print(row)

    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
