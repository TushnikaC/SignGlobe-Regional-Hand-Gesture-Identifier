import numpy as np
from utils.preprocess_utils import preprocess_image


def detect_sign(model, frame, label_encoder):
    detected_gesture = "Unknown"

    preprocessed_frame = preprocess_image(frame, IMG_SIZE=64)
    if preprocessed_frame is not None:
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        prediction = model.predict(preprocessed_frame)

        predicted_class_index = np.argmax(prediction, axis=1)[0]

        detected_gesture = label_encoder.inverse_transform([predicted_class_index])[0]

    return detected_gesture
