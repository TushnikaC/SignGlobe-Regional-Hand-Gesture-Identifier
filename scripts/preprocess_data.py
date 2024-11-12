import os
import cv2
import numpy as np

IMG_SIZE = 64


def preprocess_dataset(dataset_path, region_name):
    region_path = os.path.join(dataset_path, region_name)

    if not os.path.exists(region_path):
        raise FileNotFoundError(f"The specified region path does not exist: {region_path}")

    X = []
    y = []

    for gesture in os.listdir(region_path):
        gesture_folder = os.path.join(region_path, gesture)
        if os.path.isdir(gesture_folder):
            for img_name in os.listdir(gesture_folder):
                img_path = os.path.join(gesture_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(gesture)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = np.array(y)

    return X, y
