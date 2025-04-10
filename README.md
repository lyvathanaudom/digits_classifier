# Khmer Digit Classifier

A machine learning project for classifying handwritten Khmer digits (0-9) using various classification algorithms.

## Project Overview

This project implements a digit recognition system specifically for Khmer numerals. The system processes images of handwritten Khmer digits, converts them to a standard format, and trains machine learning models to recognize them.

### Features

- Image preprocessing (resizing, normalization)
- Binary classification (identifying specific digits)
- Multi-class classification (identifying all 10 digits)
- Model evaluation with cross-validation
- Visualization of digits and prediction results

## Project Structure

```
digits_classifier/
├── data/
│   ├── Handwritten Khmer Digit/   # Dataset of Khmer digits
│   │   ├── train/                 # Training set (sorted by digit 0-9)
│   │   ├── test/                  # Test set (sorted by digit 0-9)
│   │   └── valid/                 # Validation set (sorted by digit 0-9)
│   └── mnist/                     # Optional MNIST dataset for comparison
├── models/
│   ├── khmer_digit_classifier.py  # Main Python implementation
│   └── khmer_digit_classifier.ipynb  # Jupyter notebook version
└── README.md                      # This documentation
```

## Setup and Installation

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - NumPy
  - pandas
  - matplotlib
  - scikit-learn
  - OpenCV (cv2)
  - PIL

### Installation

1. Clone the repository or download the project files.
2. Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python pillow
```

3. Ensure your dataset is organized in the expected directory structure:
   - Each digit should have its own folder (0-9)
   - Images should be in jpg or png format

## Usage

### Running the Classifier

You can run the classifier using either the Python script or the Jupyter notebook:

```bash
python models/khmer_digit_classifier.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook models/khmer_digit_classifier.ipynb
```

### Using the Prediction Function

The model includes a function for predicting digits from new images:

```python
from models.khmer_digit_classifier import predict_digit, load_images_from_folder, rf_clf

# Load and preprocess your image
image_data = load_images_from_folder('path_to_image_folder', size=(28, 28))

# Make prediction
prediction = predict_digit(rf_clf, image_data)
print(f"Predicted digit: {prediction}")
```

## Technical Details

### Data Processing

- Images are resized to 28×28 pixels to match standard digit recognition dimensions
- Grayscale conversion to simplify the feature space
- Pixel values are normalized to the range [0, 1]
- Images are flattened to 1D vectors (784 features)

### Models

The project implements two types of classifiers:

1. **Binary Classifier**
   - Uses SGDClassifier
   - Trained to identify a specific digit (default: digit "2")
   - Evaluated using cross-validation

2. **Multi-class Classifier**
   - Uses RandomForestClassifier with 100 estimators
   - Trained to recognize all 10 digits (0-9)
   - Evaluated using cross-validation and confusion matrix

### Performance

The performance of the models is evaluated using:

- Cross-validation (3-fold)
- Accuracy metrics
- Confusion matrix for detailed error analysis

## Future Improvements

Potential enhancements for the project:

- Implement deep learning models (CNN) for improved accuracy
- Add data augmentation to increase the training set size
- Add a web interface for online digit recognition
- Expand the dataset to include more samples per digit
- Implement ensemble methods combining multiple classifier types

## License

[Add your license information here]

## Acknowledgments

- [Add credits for the dataset source]
- [Add any other acknowledgments]