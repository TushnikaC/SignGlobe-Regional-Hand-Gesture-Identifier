import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from scripts.collect_images import collect_images
from scripts.preprocess_data import preprocess_dataset
from scripts.train_model import build_model
from scripts.real_time_detection import detect_sign
from utils.model_utils import load_trained_model


def display_menu():
    print("\nSelect an option:")
    print("1. Collect Images")
    print("2. Preprocess Dataset")
    print("3. Train Model")
    print("4. Real-Time Detection")
    print("5. Exit")
    choice = input("\nEnter your choice: ")
    return choice


def main():
    dataset_path = "./dataset/"
    data_save_path = "./data/"
    models_path = "./models/"

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    while True:
        choice = display_menu()

        if choice == '1':
            region_name = input("Enter the region name (e.g., USA, India): ")
            gesture_name = input("Enter the gesture name: ")
            num_images = int(input("Enter the number of images to capture: "))
            collect_images(region_name, gesture_name, num_images)

        elif choice == '2':
            region_name = input("Enter the region name for preprocessing (e.g., USA, India): ")
            print(f"Preprocessing the dataset for region: {region_name}...")

            X, y = preprocess_dataset(dataset_path, region_name)

            np.save(os.path.join(data_save_path, f"X_{region_name}.npy"), X)
            np.save(os.path.join(data_save_path, f"y_{region_name}.npy"), y)
            print(f"Preprocessed data saved as X_{region_name}.npy and y_{region_name}.npy.")

        elif choice == '3':
            region_name = input("Enter the region name for training (e.g., USA, India): ")
            print(f"Loading preprocessed data for region: {region_name}...")

            X = np.load(os.path.join(data_save_path, f"X_{region_name}.npy"))
            y = np.load(os.path.join(data_save_path, f"y_{region_name}.npy"))

            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            np.save(os.path.join(models_path, f"classes_{region_name}.npy"), label_encoder.classes_)

            model = build_model((64, 64, 1))
            print("Training the model...")
            model.fit(X, y_encoded, epochs=10, validation_split=0.2)

            model.save(os.path.join(models_path, f"hand_sign_model_{region_name}.h5"))
            print(f"Model training completed and saved to {models_path}.")

        elif choice == '4':
            region_name = input("Enter your current region for real-time detection (e.g., USA, India): ")
            print(f"Loading trained model for region: {region_name}...")

            model_path = os.path.join(models_path, f"hand_sign_model_{region_name}.h5")

            if not os.path.exists(model_path):
                print(f"No model found for the region '{region_name}'. Please train the model first.")
                continue

            model = load_trained_model(model_path)

            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.load(os.path.join(models_path, f"classes_{region_name}.npy"))

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Unable to open the camera.")
                continue

            print("Starting real-time detection. Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to capture image from camera.")
                    break

                gesture = detect_sign(model, frame, label_encoder)

                cv2.putText(frame, f"Detected Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2)
                cv2.imshow("Hand Sign Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif choice == '5':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice, please select a valid option.")


if __name__ == "__main__":
    main()
