# 🖐️ Hand Gesture Recognition using Machine Learning

## 🎯 Goal

Recognize static American Sign Language (ASL) hand gestures in real-time using a webcam feed.

This project combines Computer Vision (OpenCV, MediaPipe) and Machine Learning (CNN + Random Forest) for gesture recognition.

## 🧠 Tech Stack

Programming Language: Python

Libraries & Frameworks:

  TensorFlow / Keras – CNN model training
  
  Scikit-learn – Random Forest classifier
  
  OpenCV – Image preprocessing & real-time video feed
  
  Streamlit + streamlit-webrtc – Web-based real-time recognition
  
  Matplotlib, Seaborn – Visualization (accuracy, confusion matrix)

## 📂 Project Structure

📁 Hand Gesture Recognition

│

├── train_model.py                # Train CNN + Random Forest models

├── real_time_recognition.py      # Streamlit app for real-time gesture recognition

│

├── sign_mnist_train.csv          # ASL training dataset

├── sign_mnist_test.csv           # ASL testing dataset

│

├── gesture_cnn_model.h5          # Saved CNN model

├── gesture_rf_model.pkl          # Saved Random Forest model

│

├── accuracy_loss_plot.png        # Accuracy and loss plot

├── confusion_matrix.png          # Confusion matrix visualization

├── classification_report.txt     # Model performance report (text)

├── classification_report.csv     # Model performance report (CSV)

│

├── american_sign_language.PNG    # ASL reference image

├── amer_sign2.png

├── amer_sign3.png

│

└── README.md                     # Project documentation


## 📊 Dataset

Dataset: Sign Language MNIST

Source: Kaggle - Sign Language MNIST - https://www.kaggle.com/datamunge/sign-language-mnist

Classes: A–Y (excluding J and Z)

Total Images: 27,455 (train) + 7,172 (test)

Each image: 28×28 grayscale

## ⚙️ How It Works

1. CNN Model (TensorFlow/Keras)

  Extracts features from 28×28 grayscale hand gesture images.
  
  Achieved 93.06% test accuracy.

2. Random Forest Classifier (Scikit-learn)

  Trained on CNN embeddings for secondary classification.
  
  Achieved ~92.93% accuracy.

3. Visualizations (Matplotlib + Seaborn)

  Accuracy/Loss curves
  
  Confusion Matrix
  
  Classification report saved automatically

4. Real-Time Recognition (Streamlit + OpenCV)

  Webcam captures hand gestures.
  
  Model predicts live ASL alphabet letter.

## 🚀 How to Run

🧩 1. Install Dependencies

pip install tensorflow scikit-learn opencv-python streamlit streamlit-webrtc seaborn matplotlib pandas

⚙️ 2. Train the Models

python train_model.py

✅ Outputs:

  gesture_cnn_model.h5
  
  gesture_rf_model.pkl
  
  accuracy_loss_plot.png, confusion_matrix.png
  
  classification_report.txt, classification_report.csv


🎥 3. Run Real-Time App

Then open the local URL shown in the terminal.

Show your hand gestures in front of the webcam — the model predicts the ASL alphabet live!

## 📈 Model Performance

| Metric     | CNN                     | Random Forest          |
| ---------- | ----------------------- | ---------------------- |
| Accuracy   | 93.06%                  | 92.93%                 |
| Input Size | 28×28                   | CNN feature embeddings |
| Labels     | 24 (A–Y excluding J, Z) | 24                     |


## 📸 Output Visuals

  accuracy_loss_plot.png – Training accuracy & loss
  
  confusion_matrix.png – Model confusion heatmap
  
  classification_report.txt – Detailed per-class metrics


<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/6cb13117-b673-4146-9b7d-f48b069db32d" />


<img width="681" height="286" alt="Figure_2" src="https://github.com/user-attachments/assets/c24fe66a-e0cf-4d10-913a-08c315b21ec3" />


## 🧩 Future Enhancements

  Add gesture tracking using MediaPipe Hands
  
  Support dynamic gestures like “J” and “Z”
  
  Deploy via Docker or Streamlit Cloud
  
  Integrate speech synthesis (text-to-speech) for recognized letters

## 👨‍💻 Author

Ittyavira C Abraham

MCA (AI), Amrita Ahead
