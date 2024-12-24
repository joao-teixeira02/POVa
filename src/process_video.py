import cv2
import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from collections import deque

def extract_features(image, extractor):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(image)
    features = extractor.predict(np.expand_dims(img, axis=0))
    return features

def process_video(video_path, cache_file, display_width=640, display_height=480):
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
        dataset_features = cache_data['features']
        dataset_images = cache_data['images']

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.8
    tracker = DeepSort(max_age=5)
    extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_time = 1/fps if fps > 0 else 1/30

    frame_skip = 1
    frame_count = 0
    feature_cache = {}
    recent_matches = deque(maxlen=5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    display_dims = (display_width, display_height)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            resized_frame = cv2.resize(frame, display_dims)
            cv2.imshow('Video', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        detection_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        results = model(detection_frame)
        detections = results.xyxy[0].cpu().numpy()

        detection_list = []
        for *box, conf, cls in detections:
            if int(cls) == 2:  # car class
                x1, y1, x2, y2 = map(lambda x: int(x * 1.25), map(int, box))
                width = x2 - x1
                height = y2 - y1
                x2 = x1 + (width // 2)  # Half of the width
                detection_list.append(([x1, y1, x2, y2], float(conf), 'car'))

        if detection_list:
            tracks = tracker.update_tracks(detection_list, frame=frame)

            matched_grid = None
            grid_width = 3
            current_grid_pos = 0
            track_ids = []  # Store track IDs for matched cars

            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                
                if track_id in feature_cache:
                    most_similar_car = feature_cache[track_id]
                else:
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    cropped_car = frame[y1:y2, x1:x2]
                    
                    if cropped_car.size > 0:
                        try:
                            detected_car_features = extract_features(cropped_car, extractor)
                            similarities = [
                                (img_file, cosine_similarity(detected_car_features, dataset_feature))
                                for img_file, dataset_feature in dataset_features[:20]
                            ]
                            most_similar_car = max(similarities, key=lambda x: x[1])
                            feature_cache[track_id] = most_similar_car
                            recent_matches.append(most_similar_car[0])
                            
                        except Exception as e:
                            continue
                
                if most_similar_car[0] in recent_matches:
                    matched_image = dataset_images[most_similar_car[0]]
                    
                    if matched_grid is None:
                        img_height = 200
                        # Add extra space for track ID label
                        matched_grid = np.zeros((img_height * ((len(tracks) - 1) // grid_width + 1),
                                              img_height * grid_width, 3), dtype=np.uint8)
                    
                    row = current_grid_pos // grid_width
                    col = current_grid_pos % grid_width
                    
                    resized_match = cv2.resize(matched_image, (img_height, img_height))
                    
                    # Add black background for track ID
                    cv2.rectangle(resized_match, (0, 0), (80, 30), (0, 0, 0), -1)
                    # Add track ID text
                    cv2.putText(
                        resized_match,
                        f"ID: {track_id}",
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    matched_grid[row * img_height:(row + 1) * img_height,
                               col * img_height:(col + 1) * img_height] = resized_match
                    
                    current_grid_pos += 1

                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            if matched_grid is not None:
                cv2.imshow('Matched Cars', matched_grid)

        resized_frame = cv2.resize(frame, display_dims)
        cv2.imshow('Video', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(feature_cache) > 30:
            feature_cache.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"
    cache_file = "dataset_features.pkl"
    
    process_video(video_path, cache_file, display_width=700, display_height=400)