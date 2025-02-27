import cv2
import numpy as np
from skimage import exposure

def preprocess_mammogram(image):
    """
    Preprocess a mammogram image for better feature extraction and classification
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to a standard size
    image = cv2.resize(image, (256, 256))
    
    # Noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image.astype(np.uint8))
    
    # Normalize pixel values
    image = image / 255.0
    
    return image

def segment_breast_region(image):
    """
    Segment the breast region from the background
    """
    # Thresholding to separate breast from background
    _, binary = cv2.threshold(
        image.astype(np.uint8), 
        0, 
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Create a mask for the largest contour (breast region)
    mask = np.zeros_like(image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        return segmented
    
    return image