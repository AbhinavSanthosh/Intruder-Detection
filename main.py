from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np
import pymongo
from datetime import datetime
import pygame
import threading

app = Flask(__name__)

model = YOLO('yolov8s.pt')

target_classes = ['person', 'cat', 'dog', 'sheep', 'cow', 'elephant', 'bear', 'pig']

client = pymongo.MongoClient("mongodb+srv://utilitico:31hrK1EvQEF23W0c@utilitico.mxknf.mongodb.net/?retryWrites=true&w=majority&appName=utilitico")
db = client["Intruder_Detection"]
collection = db["logs"]

pygame.mixer.init()
sound_file = 'static/sound.mp3'
detection_sound = pygame.mixer.Sound(sound_file)

def log_to_mongo(camera_name, label, confidence, x1, y1, x2, y2):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "camera_name": "East Boundary",
        "label": label,
        "confidence": float(confidence),
        "coordinates": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        }
    }
    collection.insert_one(log_entry)
    print(f"[{log_entry['timestamp']}] {camera_name} - Detected: {label} with confidence {log_entry['confidence']:.2f}")

def play_detection_sound():
    detection_sound.play()

def detect_and_stream():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        detection_occurred = False
        
        if not ret:
            continue

        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]

                if label in target_classes and conf >= 0.5:
                    detection_occurred = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    log_to_mongo("Single Camera", label, conf, x1, y1, x2, y2)

        if detection_occurred:
            threading.Thread(target=play_detection_sound).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
