from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import json
from datetime import datetime

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Dictionary to map feature names to cascade paths
feature_cascades = {
    "eyes": cv2.data.haarcascades + 'haarcascade_eye.xml',
    "nose": 'haar-cascade-files-master/haarcascade_mcs_nose.xml',
    "lips": 'haar-cascade-files-master/haarcascade_mcs_mouth.xml',
    "face": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
}

current_feature = "face"  # Start with detecting the face
metadata = {}  # Dictionary to hold metadata for features
capture_count = 0  # Counter to track the number of captured features

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global current_feature
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(feature_cascades[current_feature])
            features = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in features:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if current_feature == "eyes":
                    eyes_detected = len(features)
                    if eyes_detected < 2:
                        continue
                break  # Draw only the first detected feature for clarity

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next_feature', methods=['POST'])
def next_feature():
    global current_feature, capture_count
    feature_order = ["face", "eyes", "nose", "lips"]
    current_index = feature_order.index(current_feature)
    if capture_count < 4:
        current_feature = feature_order[(current_index + 1) % len(feature_order)]
    else:
        capture_count = 0  # Reset capture count after capturing all features
        current_feature = "face"  # Reset to face for the next person
    return jsonify({"status": "Success", "current_feature": current_feature})

@app.route('/capture', methods=['POST'])
def capture():
    global current_feature, capture_count, metadata
    name = request.form.get("name")

    ret, frame = cap.read()
    if not ret:
        return jsonify({"status": "Error", "message": "Failed to capture image"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(feature_cascades[current_feature])
    features = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in features:
        roi_color = frame[y:y + h, x:x + w]
        ret, buffer = cv2.imencode('.jpg', roi_color)
        image_bytes = buffer.tobytes()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists('captures'):
            os.makedirs('captures')
        filename = f"captures/{name}_{current_feature}.jpg"
        with open(filename, 'wb') as f:
            f.write(image_bytes)

        # Update metadata
        if name not in metadata:
            metadata[name] = []
        metadata[name].append({
            "feature": current_feature,
            "timestamp": timestamp,
            "filename": filename
        })

        # Save metadata to a file
        with open('metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        capture_count += 1

        return jsonify({"status": "Success", "message": f"{current_feature.capitalize()} captured successfully"})

    return jsonify({"status": "Error", "message": f"No {current_feature} detected"})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    cap.release()
    return jsonify({"status": "Success", "message": "Webcam released and server shut down"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
