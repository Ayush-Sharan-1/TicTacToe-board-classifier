import cv2
import numpy as np

def visualize_board_detection(image, corners, window_name="Board Detection", save_path=None):
    """
    Visualize the detected Tic Tac Toe board on the original image.

    Draws:
        - A green contour connecting the 4 corners
        - Red circles at each corner

    Args:
        image (np.ndarray): Original image.
        corners (np.ndarray): 4x2 array of (x, y) corner coordinates.
        window_name (str): Title for the display window.
        save_path (str): Optional path to save the visualization instead of displaying.

    Returns:
        annotated (np.ndarray): Annotated image for debugging.
    """

    annotated = image.copy()
    if corners is not None and len(corners) == 4:
        # Draw contour lines between corners
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw red circles at each corner
        for (x, y) in corners:
            cv2.circle(annotated, (int(x), int(y)), 6, (0, 0, 255), -1)
    else:
        print("Warning: Invalid or missing corners provided for visualization.")

    # Display or save
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Visualization saved to {save_path}")
    else:
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return annotated
