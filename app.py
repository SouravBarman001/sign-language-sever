
from flask import Flask, request, jsonify
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import io
from PIL import Image
import os
import time

app = Flask(__name__)

# Load the pre-trained model
model = load_model('retrained_mobnetv2_model.h5')  # Replace with the path to your h5 file

# Define labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
          'H', 'I', 'J', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z',
          'Delete', 'Nothing', 'Space']

# Initialize cvzone HandDetector
detector = HandDetector(maxHands=1)

# Preprocessing constants
offset = 20
imgSize = 300

# Directory to save preprocessed images
save_folder = "save_sign_image"
os.makedirs(save_folder, exist_ok=True)

@app.route('/classify', methods=['POST'])
def predict():
    try:
        # Check if the file is present in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Read the file as an image
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))

        # Avoid automatic orientation handling
        img = img.convert('RGB')  # Strips out EXIF orientation data
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Optional preprocessing for phone-captured images
        img = cv2.resize(img, (640, 480))  # Resize for consistent input size
        img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise

        # Detect hand
        hands, img_with_hands = detector.findHands(img, draw=True)  # Enable visualization of keypoints
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Adjust offset dynamically to stay within bounds
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(img.shape[1], x + w + offset)
            y2 = min(img.shape[0], y + h + offset)
            imgCrop = img[y1:y2, x1:x2]

            # Preprocess the cropped image
            aspectRatio = h / w
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Resize to match model input
            imgWhite = cv2.resize(imgWhite, (224, 224))

            # Save the preprocessed image for debugging
            debug_path = os.path.join(save_folder, f"{int(time.time())}.jpg")
            cv2.imwrite(debug_path, img_with_hands)

            # Normalize and reshape
            imgWhite = imgWhite.astype(np.float32) / 255.0
            imgWhite = np.expand_dims(imgWhite, axis=0)

            # Make prediction
            predictions = model.predict(imgWhite)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = labels[predicted_index]
            confidence = predictions[0][predicted_index]

            return jsonify({
                'label': predicted_label,
                'confidence': float(confidence),
                'debug_image': debug_path
            })

        else:
            return jsonify({'error': 'No hand detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    try:
        # Check if the file is present in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        # Read the file as an image
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))

        # Avoid automatic orientation handling
        img = img.convert('RGB')  # Strips out EXIF orientation data
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect hand
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Adjust offset dynamically to stay within bounds
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(img.shape[1], x + w + offset)
            y2 = min(img.shape[0], y + h + offset)
            imgCrop = img[y1:y2, x1:x2]

            # Preprocess the cropped image
            aspectRatio = h / w
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Resize to match model input
            imgWhite = cv2.resize(imgWhite, (224, 224))

            # Save the preprocessed image
            save_path = os.path.join(save_folder, f"{int(time.time())}.jpg")
            cv2.imwrite(save_path, imgWhite)

            # Prepare for prediction
            imgWhite = imgWhite.astype(np.float32) / 255.0  # Normalize
            imgWhite = np.expand_dims(imgWhite, axis=0)

            # Make prediction
            predictions = model.predict(imgWhite)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = labels[predicted_index]
            confidence = predictions[0][predicted_index]

            return jsonify({
                'label': predicted_label,
                'confidence': float(confidence),
                'saved_image': save_path
            })

        else:
            return jsonify({'error': 'No hand detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify({
        'accuracy': "Model accuracy not available through this endpoint"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)   
