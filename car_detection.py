import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

dataset_path = "data/images/image"
video_path = "video.mp4"

print("Loading dataset...")
dataset_features = []
extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(image)
    features = extractor.predict(np.expand_dims(img, axis=0))
    return features

# Load dataset images
acc = 0
while acc < 100:
    acc = 0
    for img_file in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            features = extract_features(img)
            dataset_features.append((img_file, features))
            acc += 1
            if acc == 10:
                break

print(f"Loaded {len(dataset_features)} images from the dataset.")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
tracker = DeepSort(max_age=5)

print("Processing video...")
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    detection_list = []
    for *box, conf, cls in detections:
        if int(cls) == 2:  # class 2 is car in COCO dataset
            x1, y1, x2, y2 = map(int, box)
            # Convert to format expected by DeepSort: [[x1, y1, x2, y2], confidence_score]
            detection_list.append(([x1, y1, x2, y2], float(conf), 'car'))

    # Update tracker with the correct detection format
    if detection_list:
        tracks = tracker.update_tracks(detection_list, frame=frame)

        # Draw tracking results
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            cropped_car = frame[y1:y2, x1:x2]
            
            # Only process if we have a valid crop
            if cropped_car.size > 0:
                try:
                    detected_car_features = extract_features(cropped_car)
                    similarities = [
                        (img_file, cosine_similarity(detected_car_features, dataset_feature))
                        for img_file, dataset_feature in dataset_features
                    ]
                    most_similar_car = max(similarities, key=lambda x: x[1])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {track_id}, Match: {most_similar_car[0]}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                except Exception as e:
                    print(f"Error processing track {track_id}: {e}")
                    continue

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Processing complete.")