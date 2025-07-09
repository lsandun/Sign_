import tensorflow as tf
import numpy as np
import os
import cv2

# Function to load and preprocess an image
def preprocess_image(image_path, target_size=(64, 64)):
    if isinstance(image_path, str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # If image_path is already a numpy array
        img = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY) if len(image_path.shape) == 3 else image_path
    
    img = cv2.resize(img, target_size)  # Ensure correct resizing
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Function to extract the student name from the file path
def get_student_name(image_path):
    return os.path.basename(os.path.dirname(os.path.dirname(image_path)))

# Function to verify signatures
def verify_signature(model, ref_image_path, test_image_path, threshold=0.5):
    ref_image = preprocess_image(ref_image_path)
    test_image = preprocess_image(test_image_path)

    # Add batch dimension
    ref_image = np.expand_dims(ref_image, axis=0)
    test_image = np.expand_dims(test_image, axis=0)

    similarity = model.predict([ref_image, test_image])[0][0]
    is_genuine = similarity > threshold
    
    # Get student name if ref_image_path is a string path
    student_name = get_student_name(ref_image_path) if isinstance(ref_image_path, str) else "Unknown"
    
    return student_name, is_genuine, similarity * 100