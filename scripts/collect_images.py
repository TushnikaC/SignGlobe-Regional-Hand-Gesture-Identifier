import os
import cv2


def collect_images(region_name, gesture_name, num_images):
    gesture_path = os.path.join("./data/dataset/", region_name, gesture_name)

    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

    cap = cv2.VideoCapture(0)
    print(f"Collecting images for region: {region_name}, gesture: {gesture_name} (Press 'c' to capture, 'q' to quit)")

    img_count = 0
    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img_path = os.path.join(gesture_path, f"{img_count + 1}.jpg")
            cv2.imwrite(img_path, frame)
            img_count += 1
            print(f"Captured {img_count}/{num_images} images")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting image capture.")
