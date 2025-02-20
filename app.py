# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np

# app = Flask(__name__)

# # Load Pre-trained Models
# faceNet = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
# ageNet = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
# genderNet = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")

# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/detect', methods=['POST'])
# def detect():
#     try:
#         file = request.files['file']
#         img = np.frombuffer(file.read(), np.uint8)
#         frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
#         faceNet.setInput(blob)
#         detections = faceNet.forward()

#         if len(detections) == 0:
#             return jsonify({"error": "No face detected!"})

#         genderNet.setInput(blob)
#         genderPreds = genderNet.forward()
#         gender = genderList[genderPreds[0].argmax()]

#         ageNet.setInput(blob)
#         agePreds = ageNet.forward()
#         age = ageList[agePreds[0].argmax()][1:-1]

#         return jsonify({"gender": gender, "age": age})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

# Load Pre-trained Models
faceNet = cv2.dnn.readNet("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
genderNet = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Confidence threshold for face detection
CONFIDENCE_THRESHOLD = 0.7
# Padding around the detected face for age/gender analysis
PADDING = 20

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['file']
        img_data = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image!"})
        
        # Create blob from image for face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        h, w = frame.shape[:2]
        face_found = False
        face_box = None

        # detections shape is usually [1, 1, N, 7]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                # Get bounding box
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                face_box = [x1, y1, x2, y2]
                face_found = True
                break  # using the first detected face

        if not face_found or face_box is None:
            return jsonify({"error": "No face detected!"})

        # Crop the detected face region with padding (ensure within image bounds)
        x1, y1, x2, y2 = face_box
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w - 1, x2 + PADDING)
        y2 = min(h - 1, y2 + PADDING)
        face = frame[y1:y2, x1:x2]

        # Prepare blob for age and gender prediction
        # Age/gender models usually expect a 227x227 face image
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Predict Gender
        genderNet.setInput(face_blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict Age
        ageNet.setInput(face_blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        # Return results as JSON
        return jsonify({"gender": gender, "age": age[1:-1]})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

