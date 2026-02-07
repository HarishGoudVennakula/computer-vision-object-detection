import cv2
import numpy as np


def start_camera(camera_index=0):
    """
    Initialize the webcam.
    camera_index: 0 = default camera
    Returns the VideoCapture object.
    """
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError("Cannot access the camera")
    return cam


def preprocess_frame(frame):
    """
    Convert the frame to grayscale, apply Gaussian blur, and adaptive thresholding.
    Returns a binary image suitable for contour detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    return binary


def detect_objects(binary_img, output_frame, min_area=800):
    """
    Detect contours in the binary image and draw bounding boxes on output_frame.
    Returns the number of detected objects.
    """
    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        count += 1
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display area above the box
        cv2.putText(
            output_frame,
            f"Size: {int(area)}",
            (x, max(y - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

    return count


def main():
    # Initialize webcam
    camera = start_camera()

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Standardize frame size
        frame = cv2.resize(frame, (640, 480))

        # Process frame for detection
        binary_frame = preprocess_frame(frame)
        objects_detected = detect_objects(binary_frame, frame)

        # Display object count
        cv2.putText(
            frame,
            f"Objects: {objects_detected}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Show frames
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Processed View", binary_frame)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
