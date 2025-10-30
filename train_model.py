import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN custom operations

# 📥 Load Dataset
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

y_train = train_df['label']
X_train = train_df.drop('label', axis=1)
y_test = test_df['label']
X_test = test_df.drop('label', axis=1)

# Normalize and reshape
X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.values.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = to_categorical(y_train, num_classes=26)
y_test_cat = to_categorical(y_test, num_classes=26)

# 🧠 Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train Model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=10,
    batch_size=128
)

# 💾 Save Trained Model
model.save("gesture_cnn_model.h5")
print("\n✅ Model saved as gesture_cnn_model.h5\n")


# Train Random Forest classifier for comparison 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Convert CNN features into embeddings (flattened output)
X_features = model.predict(X_train).reshape(X_train.shape[0], -1)
X_test_features = model.predict(X_test).reshape(X_test.shape[0], -1)

# Train RF model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_features, y_train)
y_pred_rf = rf.predict(X_test_features)

# Evaluate RF
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# Save RF model
with open('gesture_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)


# 📊 Accuracy & Loss Graph (Save)
plt.figure(figsize=(12,5))

# Accuracy Plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_loss_plot.png", dpi=300)
print("📈 Saved: accuracy_loss_plot.png")
plt.show()

# 🔍 Confusion Matrix & Report (Save)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[chr(i) for i in range(65,91)],
            yticklabels=[chr(i) for i in range(65,91)])
plt.title('Confusion Matrix - ASL Gesture Recognition')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig("confusion_matrix.png", dpi=300)
print("📊 Saved: confusion_matrix.png")
plt.show()

print("\n🧾 Classification Report:")
gesture_labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
report = classification_report(y_test, y_pred_classes, target_names=gesture_labels, digits=4)
print(report)

# Save classification report to text file 
with open("classification_report.txt", "w") as f:
    f.write("Classification Report (CNN Model)\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)

# Also save as CSV 
report_dict = classification_report(y_test, y_pred_classes, target_names=gesture_labels, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("classification_report.csv", index=True)

print("🧩 Saved: classification_report.txt and classification_report.csv")
