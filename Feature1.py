import cv2
import numpy as np

# Initializing the ORB Feature Detector
MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000)

# Preparing the FLANN Based matcher
index_params = dict(algorithm=1, trees=3)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Function for Loading input image and Keypoints
def load_input():
    #path of your image
    input_image = cv2.imread("path")
    if input_image is None:
        raise FileNotFoundError("The image file 't1.jpg' was not found.")
    input_image = cv2.resize(input_image, (400, 550), interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # find the keypoints with ORB
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    return gray_image, keypoints, descriptors

# Function for Computing Matches between the train and query descriptors
def compute_matches(descriptors_input, descriptors_output):
    if descriptors_output is not None and descriptors_input is not None and len(descriptors_output) != 0 and len(descriptors_input) != 0:
        matches = flann.knnMatch(np.asarray(descriptors_input, np.float32), np.asarray(descriptors_output, np.float32), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.68 * n.distance:
                good.append([m])
        return good
    else:
        return None

# Main Working Logic
if __name__ == '__main__':
    # Getting Information form the Input image
    input_image, input_keypoints, input_descriptors = load_input()

    # Getting camera ready
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Condition Check for error escaping
        if len(input_keypoints) < MIN_MATCHES:
            continue
        
        # Resizing input image for fast computation
        frame = cv2.resize(frame, (700, 600))
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Computing and matching the Keypoints of Input image and query Image
        output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
        if output_descriptors is not None:
            matches = compute_matches(input_descriptors, output_descriptors)
            if matches is not None:
                output_final = cv2.drawMatchesKnn(input_image, input_keypoints, frame, output_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow('Final Output', output_final)
            else:
                cv2.imshow('Final Output', frame)
        else:
            cv2.imshow('Final Output', frame)
        
        key = cv2.waitKey(5)
        if key == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
