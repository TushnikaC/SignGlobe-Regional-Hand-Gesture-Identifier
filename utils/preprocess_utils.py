import cv2
import numpy as np


def preprocess_image(frame, IMG_SIZE=64):
    try:
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        normalized_frame = gray_frame / 255.0

        preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)

        return preprocessed_frame
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None