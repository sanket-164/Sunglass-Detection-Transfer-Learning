import cv2
import numpy as np
import keras

model = keras.models.load_model("model.h5")

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the frame to 256x256
    resized_frame = cv2.resize(frame, (256, 256))

    # Use the below kernels if you do not have high resolution you van reduce or aplly more filters
    kernel_1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel_2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    resized_frame = cv2.filter2D(resized_frame, -1, kernel_1)

    # Apply more kernels here

    # Normalizing the image
    resized_frame = resized_frame / 255.0

    # Convert to a numpy array of shape (256, 256, 3)
    pixel_data = np.array(resized_frame)

    # Predict using the model
    print(model.predict(pixel_data.reshape((1, 256, 256, 3))))

    if model.predict(pixel_data.reshape((1, 256, 256, 3))) >= 0.5:
        cv2.putText(
            frame,
            "Sunglasses Detected",  # Text to display
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            (0, 0, 255),  # Font color in BGR
            2,  # Thickness
            cv2.LINE_AA,
        )  # Line type

    cv2.imshow("Camera Feed", frame)

    # Press 'q' to quit without saving
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
