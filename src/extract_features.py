import cv2
import os
import numpy as np
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from datetime import datetime

def extract_features(image, extractor):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(image)
    features = extractor.predict(np.expand_dims(img, axis=0))
    return features

def process_dataset(dataset_path, cache_file, max_images=100):
    print("Initializing feature extraction...")
    start_time = datetime.now()
    
    # Initialize ResNet model
    extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    dataset_features = []
    dataset_images = {}
    acc = 0

    # Process images
    for img_file in os.listdir(dataset_path):
        if acc >= max_images:
            break
            
        img_path = os.path.join(dataset_path, img_file)
        img = cv2.imread(img_path)
        
        if img is not None:
            print(f"Processing image {acc + 1}/{max_images}: {img_file}")
            features = extract_features(img, extractor)
            dataset_features.append((img_file, features))
            dataset_images[img_file] = cv2.resize(img, (224, 224))
            acc += 1

    # Save to cache
    print("Saving features to cache...")
    cache_data = {
        'features': dataset_features,
        'images': dataset_images
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    end_time = datetime.now()
    print(f"Feature extraction complete. Processed {acc} images.")
    print(f"Total processing time: {end_time - start_time}")

if __name__ == "__main__":
    dataset_path = "data/images"
    cache_file = "dataset_features.pkl"
    max_images = 100
    
    process_dataset(dataset_path, cache_file, max_images)