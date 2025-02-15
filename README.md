# Sunglass Detection

This project implements a real-time sunglass detection system using TensorFlow, Keras, NumPy, and OpenCV, with transfer learning for feature extraction using the VGG16 pre-trained model. The system captures live video from the webcam, preprocesses each frame, and uses a deep learning model built on top of VGG16 to predict whether the person in the frame is wearing sunglasses.

## **Technologies Used**
- **TensorFlow/Keras**: For building and loading the deep learning model.
- **OpenCV**: For real-time video capture and image processing.
- **NumPy**: For handling arrays and numerical operations.
- **VGG16**: Pre-trained model for feature extraction to improve accuracy and reduce training time.

## **Project Structure**
- `app.py`: Main script to capture video, preprocess frames, and make predictions.
- `model.keras`: Trained deep learning model for sunglass detection using VGG16 with keras format.
- `model.h5`: Trained deep learning model for sunglass detection using VGG16 with h5 format.
- `requirements.txt`: List of required packages.

## **Usage**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sanket-164/Sunglass-Detection-Transfer-Learning.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd sunglass-detection
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project:**
   ```bash
   python app.py
   ```

## **Model Training**
- Utilizes the VGG16 model for feature extraction.
- Top layers are fine-tuned for sunglass detection.
- Augmentation techniques like flipping, rotation, and scaling are applied.
- Optimizer: Adam, Loss: Categorical Crossentropy.

## **Preprocessing Steps**
- Captured frames are resized to 256x256.
- Sharpening filters are applied for better feature extraction.
- Pixel values are normalized to [0, 1].

## **Output**
- Displays live webcam feed with predictions.
- Prints prediction probabilities in the console.
- Shows Detection of sunglasses in live frame.

## **Troubleshooting**
- Ensure good lighting for accurate predictions.
- Increase camera resolution in the code if the quality is poor.
- Use a sharpening filter to enhance image clarity.

## **License**
This project is licensed under the MIT License.
