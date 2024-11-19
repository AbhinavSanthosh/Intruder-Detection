from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import pymongo
from datetime import datetime
import pygame
import threading
from twilio.rest import Client

load_dotenv()

app = Flask(__name__)

model = YOLO('yolov8s.pt')
target_classes = ['person', 'cat', 'dog', 'sheep', 'cow', 'elephant', 'bear', 'pig']

mongo_url = os.getenv("MONGO_URL")
client = pymongo.MongoClient(mongo_url)
db = client["Intruder_Detection"]
collection = db["logs"]

pygame.mixer.init()
sound_file = 'static/sound.mp3'
detection_sound = pygame.mixer.Sound(sound_file)

sid = os.getenv("SID")
auth_token = os.getenv("AUTH_TOKEN")
whatsapp_number = os.getenv("WHATSAPP_NUMBER")
recipient_whatsapp_number = os.getenv("RECIPIENT_WHATSAPP_NUMBER")

twilio_client = Client(sid, auth_token)

recording = False
video_writer = None
video_dir = "recordings"
os.makedirs(video_dir, exist_ok=True)

def log_to_mongo(camera_name, label, confidence, x1, y1, x2, y2):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "camera_name": camera_name,
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

def play_detection_sound():
    detection_sound.play()

def send_whatsapp_message(label, confidence):
    message = f"Intruder Alert! Detected: {label} with confidence: {confidence:.2f}"
    try:
        twilio_client.messages.create(
            body=message,
            from_=f'whatsapp:{whatsapp_number}',
            to=f'whatsapp:{recipient_whatsapp_number}'
        )
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

def start_video_recording(frame, fps):
    global video_writer, recording
    if not recording:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = os.path.join(video_dir, f"Intruder_{timestamp}.mp4")
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        recording = True

def stop_video_recording():
    global video_writer, recording
    if recording:
        video_writer.release()
        video_writer = None
        recording = False

def detect_and_stream():
    global video_writer, recording
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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
                    log_to_mongo("East Camera", label, conf, x1, y1, x2, y2)
                    send_whatsapp_message(label, conf)

        if detection_occurred:
            threading.Thread(target=play_detection_sound).start()
            if not recording:
                start_video_recording(frame, fps)
        elif recording:
            stop_video_recording()

        if recording and video_writer:
            video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    if recording:
        stop_video_recording()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
