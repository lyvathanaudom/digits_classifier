import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import cv2

# Define paths
DATA = 'c:/Users/Asus/OneDrive/Desktop/projects/personal-projects/ai-projects/digits_classifier/data/Handwritten Khmer Digit'
TRAIN_DIR = os.path.join(DATA, 'train')
TEST_DIR = os.path.join(DATA, 'test')
VALID_DIR = os.path.join(DATA, 'valid')

def load_images_from_folder(folder, size=(28, 28)):
    """
    Load all images from a folder and resize them to the specified size.
    Returns a tuple of (images, labels)
    """
    images = []
    labels = []
    
    # Iterate over digit folders (0-9)
    for digit in range(10):
        digit_folder = os.path.join(folder, str(digit))
        
        # Skip if folder doesn't exist
        if not os.path.exists(digit_folder):
            continue
            
        # Iterate over all files in the digit folder
        for filename in os.listdir(digit_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(digit_folder, filename)
                try:
                    # Read the image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    # Resize to match MNIST format
                    img = cv2.resize(img, size)
                    
                    # Normalize pixel values to be between 0 and 1
                    img = img / 255.0
                    
                    # Flatten the image to 1D array (28x28 -> 784)
                    img_flat = img.flatten()
                    
                    # Add to our dataset
                    images.append(img_flat)
                    labels.append(str(digit))  # Convert digit to string to match MNIST format
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

# Load datasets
print("Loading training data...")
X_train_data, y_train_data = load_images_from_folder(TRAIN_DIR)
print("Loading testing data...")
X_test_data, y_test_data = load_images_from_folder(TEST_DIR)
print("Loading validation data...")
X_valid_data, y_valid_data = load_images_from_folder(VALID_DIR)

# Combine training and validation data for full training set
X_train = np.vstack((X_train_data, X_valid_data))
y_train = np.concatenate((y_train_data, y_valid_data))
X_test = X_test_data
y_test = y_test_data

print(f"Training set: {X_train.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# Shuffle the training data
shuffle_index = np.random.permutation(len(X_train))
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# Visualize a sample digit
def plot_digit(data, label=None):
    """Plot a single digit with its label"""
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    if label is not None:
        plt.title(f"Digit: {label}")
    plt.show()

# Display a random sample from the training data
sample_idx = np.random.randint(0, len(X_train))
print(f"Sample digit: {y_train[sample_idx]}")
plot_digit(X_train[sample_idx], y_train[sample_idx])

# Binary classification example (is_digit_2)
print("Training binary classifier for digit 2...")
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')

# Train SGD Classifier for binary classification (digit 2 vs not digit 2)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_2)

# Cross-validation
cv_scores = cross_val_score(sgd_clf, X_train, y_train_2, cv=3, scoring="accuracy")
print("Cross-validation scores (binary):", cv_scores)
print("Cross-validation mean accuracy (binary):", cv_scores.mean())

# Multi-class classification
print("Training multi-class classifier...")
# Use Random Forest which tends to work well for image data
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Cross-validation for multi-class
cv_scores_multi = cross_val_score(rf_clf, X_train, y_train, cv=3, scoring="accuracy")
print("Cross-validation scores (multi-class):", cv_scores_multi)
print("Cross-validation mean accuracy (multi-class):", cv_scores_multi.mean())

# Evaluate on test set
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Function to predict a single image
def predict_digit(classifier, image_data):
    """Predict the digit for a single image"""
    # Ensure the image is flattened
    if image_data.ndim > 1:
        image_data = image_data.flatten()
        
    # Reshape for prediction (classifier expects 2D array)
    image_reshaped = image_data.reshape(1, -1)
    
    # Make prediction
    prediction = classifier.predict(image_reshaped)[0]
    return prediction

# Demo prediction on a test image
test_idx = np.random.randint(0, len(X_test))
test_img = X_test[test_idx]
true_label = y_test[test_idx]
predicted_label = predict_digit(rf_clf, test_img)

print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")
plot_digit(test_img, f"True: {true_label}, Pred: {predicted_label}")