import numpy as np
import pywt
import cv2
from skimage.feature import local_binary_pattern

def wavelet_transform(image, wavelet='db1', level=3):
    """
    Extract wavelet features from an image.
    This is particularly relevant to microwave imaging techniques.
    """
    # Apply wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Extract features from coefficients
    features = []
    for i in range(level):
        # For each level, extract statistical features from detail coefficients
        for detail_coeff in coeffs[i+1]:
            features.append(np.mean(detail_coeff))
            features.append(np.std(detail_coeff))
            features.append(np.max(detail_coeff))
            features.append(np.min(detail_coeff))
    
    # Extract features from approximation coefficient
    features.append(np.mean(coeffs[0]))
    features.append(np.std(coeffs[0]))
    features.append(np.max(coeffs[0]))
    features.append(np.min(coeffs[0]))
    
    return np.array(features)

def extract_lbp_features(image, radius=3, n_points=24):
    """
    Extract Local Binary Pattern features for texture analysis
    """
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
    return hist

def extract_all_features(image):
    """
    Combine multiple feature extraction techniques
    """
    # Resize image for consistency
    image = cv2.resize(image, (256, 256))
    
    # Apply preprocessing to enhance features
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Extract different types of features
    wavelet_features = wavelet_transform(image)
    lbp_features = extract_lbp_features(image)
    
    # Combine features
    combined_features = np.concatenate([wavelet_features, lbp_features])
    
    return combined_features