import cv2
import numpy as np
import tensorflow as tf

def main():
    print("Loading model...")
    try:
        model = tf.keras.models.load_model('gesture_model.h5')
    except Exception as e:
        print("Model not found. Please train the model first by running train_model.py")
        return

    # Configuration
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    class_names = [str(i) for i in range(10)]

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam... Press 'q' to quit.")

    # Define Region of Interest (ROI) coordinates
    x1, y1, x2, y2 = 100, 100, 400, 400

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for natural viewing
        frame = cv2.flip(frame, 1)

        # Draw ROI rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Place hand in box", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Preprocess ROI for prediction
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(roi_resized)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # Predict
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        # Display result
        text = f"{predicted_class}"
        cv2.putText(frame, text, (x2+20, y1+(y2-y1)//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        cv2.putText(frame, f"Conf: {confidence:.2f}%", (x2+20, y1+(y2-y1)//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
