# Hand Gesture Recognition Project

This project predicts hand gestures (digits 0-9) using a Convolutional Neural Network (CNN).

## Features
- **Automated Data Collection**: Downloads a public sign language dataset.
- **Model Training**: Preprocesses and augments images, and trains a CNN model.
- **Streamlit Web UI**: Allows users to upload a hand image or take a picture via webcam for prediction.
- **Real-time Prediction**: Includes an OpenCV script for live predictions via your webcam.

## File Structure
- `requirements.txt`: Python dependencies
- `data_collection.py`: Script to download the dataset
- `train_model.py`: CNN model building and training script
- `app.py`: Streamlit application
- `realtime_webcam.py`: Real-time OpenCV webcam application
- `gesture_model.h5`: The trained model (generated after training)
- `dataset_repo/`: Downloaded dataset directory (generated after data collection)

## Step-by-Step Run Instructions

1. **Install Dependencies**
   Make sure you have Python installed. Open your terminal or command prompt and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**
   Run the data collection script to download the "Sign Language Digits Dataset":
   ```bash
   python data_collection.py
   ```

3. **Train the Model**
   Train the CNN model. This will read the dataset, train for 15 epochs, and save `gesture_model.h5`:
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit App**
   Start the web application to use the model:
   ```bash
   streamlit run app.py
   ```
   Open your browser at the provided URL (usually `http://localhost:8501`). You can upload an image or use your webcam to take a picture, and it will output ONLY the predicted digit.

5. **(Bonus) Run Real-time Webcam Prediction**
   If you want to try live predictions directly via a windowed application:
   ```bash
   python realtime_webcam.py
   ```
   Place your hand in the blue box to see the prediction. Press `q` to quit.
