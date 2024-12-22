import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

dataset_path = "data/images"
video_path = "video.mp4"

print("Loading dataset...")
dataset_features = []
extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(image)
    features = extractor.predict(np.expand_dims(img, axis=0))
    return features

acc = 0

while acc < 10:
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
tracker = DeepSort()

print("Processing video...")
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    bbox_xywh = []
    confidences = []
    for *box, conf, cls in detections:
        if int(cls) == 2:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            x_center, y_center = x1 + w // 2, y1 + h // 2
            bbox_xywh.append([x_center, y_center, w, h])
            confidences.append(float(conf))

    bbox_xywh = np.array(bbox_xywh)
    confidences = np.array(confidences)

    print("BBox (XYWH):", bbox_xywh)
    print("BBox (XYWH) Shape:", bbox_xywh.shape)
    print("Confidences:", confidences)
    print("Confidences Shape:", confidences.shape)

    # TRACKER STILL NOT WORKING
    outputs = tracker.update_tracks(bbox_xywh, confidences, frame)

    for track in outputs:
        track_id = track.track_id
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        cropped_car = frame[y1:y2, x1:x2]

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

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
