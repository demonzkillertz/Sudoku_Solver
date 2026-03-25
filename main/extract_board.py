import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import imutils
import argparse
import sys

# Model parameters
input_size = 48
classes = np.arange(0, 10)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_perspective(img, location, height=900, width=900):
    """Takes an image and location of interested region.
       And returns only the selected region with a perspective transformation"""
    pts1 = order_points(location.reshape(4, 2))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Advanced Preprocessing for camera images
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    
    keypoints = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    
    # Finds rectangular contour
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            location = approx
            break
            
    if location is None:
        print("Warning: Advanced thresholding failed, falling back to Canny edge detection.")
        bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
        edged = cv2.Canny(bfilter, 30, 180)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours  = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 15, True)
            if len(approx) == 4:
                location = approx
                break

    if location is None:
        print("Error: Could not find a Sudoku board contour in the image.")
        return None, None
        
    result = get_perspective(img, location)
    return result, location

def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells. 
       each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board, 9)
    boxes = []
    empty_flags = []
    
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            # 1. Crop heavily to remove grid line artifacts (crop 15% from all sides)
            h, w = box.shape
            pad_h, pad_w = int(h * 0.15), int(w * 0.15)
            cropped = box[pad_h:h-pad_h, pad_w:w-pad_w]
            
            # Blur out minor paper texture noise
            blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
            
            # Use Adaptive Threshold with a large constant strictly suppressing shadows and gradients.
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)
            
            # Morphological opening to kill leftover speckles
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours in the cell
            cnts, _ = cv2.findContours(thresh_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            is_empty = True
            digit_rect = None
            
            if cnts:
                valid_cnts = []
                for c in cnts:
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    area = cv2.contourArea(c)
                    
                    valid_height = cropped.shape[0] * 0.85 > ch > cropped.shape[0] * 0.20
                    valid_width = cropped.shape[1] * 0.85 > cw > cropped.shape[1] * 0.05
                    valid_area = area > 10
                    touching_edge = (cx == 0) or (cy == 0) or (cx + cw >= cropped.shape[1]-1) or (cy + ch >= cropped.shape[0]-1)
                    
                    if valid_height and valid_width and valid_area and not touching_edge:
                        valid_cnts.append(c)
                        
                if valid_cnts:
                    is_empty = False
                    # Get unified bounding box for the digit
                    x_min = min([cv2.boundingRect(c)[0] for c in valid_cnts])
                    y_min = min([cv2.boundingRect(c)[1] for c in valid_cnts])
                    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_cnts])
                    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_cnts])
                    digit_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                        
            if is_empty or digit_rect is None:
                empty_flags.append(True)
                res = np.zeros((input_size, input_size), dtype=np.float32)
            else:
                empty_flags.append(False)
                
                # Extract clean bounding box of the digit from the original NOT inverted cropped image
                # Wait, removing shadows completely helps OCR trained on clean lines. 
                # Let's extract the digit from the thresh_clean (which is white digit on black bg)
                dx, dy, dw, dh = digit_rect
                digit_mask = thresh_clean[dy:dy+dh, dx:dx+dw]
                
                # Invert back to black digit on white background (what standard OCR expects)
                digit_img = cv2.bitwise_not(digit_mask)
                
                # Center the digit in a 48x48 white background
                white_bg = np.full((input_size, input_size), 255, dtype=np.uint8)
                
                # Keep aspect ratio when scaling the digit to fit nicely within 48x48 (leave a margin)
                margin = int(input_size * 0.2)
                tgt_size = input_size - 2 * margin
                
                scale = min(tgt_size / dw, tgt_size / dh)
                new_w, new_h = int(dw * scale), int(dh * scale)
                
                resized_digit = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Calculate paste positioning
                start_x = (input_size - new_w) // 2
                start_y = (input_size - new_h) // 2
                
                white_bg[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit
                
                # Normalize 0-1
                res = white_bg / 255.0
                
            boxes.append(res)
    return boxes, empty_flags

def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] != 0:
                cv2.putText(img, str(numbers[(j*9)+i]), 
                            (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), 
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2, cv2.LINE_AA)
    return img

def main():
    parser = argparse.ArgumentParser(description="Extract board and evaluate OCR models (H5 vs TFLite)")
    parser.add_argument("--image", type=str, default="test_sudokuu.jpg", help="Path to the test image")
    args = parser.parse_args()

    # 1. Read the image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at path: {args.image}")
        sys.exit(1)

    # 2. Extract Board
    print("Finding valid board contour...")
    board, location = find_board(img)
    if board is None:
        sys.exit(1)

    print("Board extracted successfully.")

    # Show the original extraction
    cv2.imshow("Extracted Board", board)
    cv2.waitKey(1) # Refresh window

    # 3. Process board into ROIs
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois, empty_flags = split_boxes(gray)
    rois_np = np.array(rois).reshape(-1, input_size, input_size, 1).astype(np.float32)

    # 4. Load Models and Predict
    # -- H5 Model --
    print("Loading H5 Model...")
    h5_model = load_model("model-OCR.h5")
    print("Running inference on H5 model...")
    h5_predsRaw = h5_model.predict(rois_np)
    
    h5_preds = []
    for i, p in enumerate(h5_predsRaw):
        if empty_flags[i]:
            h5_preds.append(0)
        else:
            h5_preds.append(classes[np.argmax(p)])
    
    # -- TFLite Model --
    print("Loading TFLite Model...")
    interpreter = tf.lite.Interpreter(model_path="model-OCR.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # TFLite might expect float32
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    tflite_preds = []
    print("Running inference on TFLite model...")
    for i, roi in enumerate(rois_np):
        if empty_flags[i]:
            tflite_preds.append(0)
            continue
            
        roi_input = np.expand_dims(roi, axis=0) # Shape: (1, 48, 48, 1)
        interpreter.set_tensor(input_index, roi_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        tflite_preds.append(classes[np.argmax(output)])

    # 5. Output Results
    print("\n--- Model Evaluation Results ---")
    
    print("\nH5 Model Output:")
    h5_matrix = np.array(h5_preds).reshape(9, 9)
    print(h5_matrix)

    print("\nTFLite Model Output:")
    tflite_matrix = np.array(tflite_preds).reshape(9, 9)
    print(tflite_matrix)
    
    print("\nDifferences (H5 vs TFLite):")
    diffs = 0
    for r in range(9):
        for c in range(9):
            if h5_matrix[r][c] != tflite_matrix[r][c]:
                print(f"Row {r} Col {c} : H5 = {h5_matrix[r][c]} | TFLite = {tflite_matrix[r][c]}")
                diffs += 1
    if diffs == 0:
        print("No differences found. Both models output the exact same predictions.")

    # 6. Display visually
    h5_board_render = board.copy()
    displayNumbers(h5_board_render, h5_preds, color=(0, 255, 0))
    cv2.imshow("H5 Predictions", h5_board_render)

    tflite_board_render = board.copy()
    displayNumbers(tflite_board_render, tflite_preds, color=(255, 0, 0))
    cv2.imshow("TFLite Predictions", tflite_board_render)
    
    print("\nPress any key in the image windows to close and exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()