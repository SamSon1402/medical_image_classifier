import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import torch
import pywt
from skimage.feature import local_binary_pattern
from skimage import exposure

def plot_image_grid(images, titles=None, cmaps='gray', figsize=(10, 10), grid_shape=None):
    """
    Plot a grid of images
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        cmaps: Colormap(s) to use for the images
        figsize: Figure size
        grid_shape: Tuple specifying grid dimensions (rows, cols)
        
    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    
    if grid_shape is None:
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))
    else:
        n_rows, n_cols = grid_shape
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot each image
    for i, ax in enumerate(axes.flatten()):
        if i < n_images:
            # Handle different image dimensions and channels
            img = images[i]
            if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            
            if len(img.shape) == 3 and img.shape[0] == 1:  # Single channel
                img = img[0]
            
            # Normalize if needed
            if img.max() > 1.0:
                img = img / 255.0
                
            ax.imshow(img, cmap=cmaps if isinstance(cmaps, str) else cmaps[i])
            
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])
            
            ax.axis('off')
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    return fig

def visualize_preprocessing_steps(original_image, preprocessed_stages):
    """
    Visualize the preprocessing pipeline steps
    
    Args:
        original_image: Original input image
        preprocessed_stages: Dictionary of images at different preprocessing stages
        
    Returns:
        Matplotlib figure
    """
    # Collect all images
    images = [original_image]
    titles = ['Original']
    
    for stage_name, stage_image in preprocessed_stages.items():
        images.append(stage_image)
        titles.append(stage_name)
    
    # Plot grid
    fig = plot_image_grid(images, titles, cmaps='gray')
    plt.suptitle('Preprocessing Pipeline Visualization', fontsize=16)
    
    return fig

def visualize_feature_extraction(image, method='wavelet'):
    """
    Visualize feature extraction methods
    
    Args:
        image: Input image
        method: Feature extraction method ('wavelet', 'lbp', or 'histogram')
        
    Returns:
        Matplotlib figure
    """
    if method == 'wavelet':
        return visualize_wavelet_decomposition(image)
    elif method == 'lbp':
        return visualize_lbp_features(image)
    elif method == 'histogram':
        return visualize_histogram_features(image)
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

def visualize_wavelet_decomposition(image, wavelet='db1', level=2):
    """
    Visualize wavelet decomposition of an image
    
    Args:
        image: Input image
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Matplotlib figure
    """
    # Ensure image is 2D
    if len(image.shape) > 2:
        if image.shape[0] == 1:  # CHW format with 1 channel
            image = image[0]
        elif image.shape[2] == 1:  # HWC format with 1 channel
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Create a figure to display the decomposition
    fig, axes = plt.subplots(level + 1, 3, figsize=(12, 4 * (level + 1)))
    
    # Plot approximation coefficients
    axes[0, 0].imshow(coeffs[0], cmap='viridis')
    axes[0, 0].set_title('Approximation')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    # Plot detail coefficients
    for i in range(level):
        # Horizontal detail
        axes[i+1, 0].imshow(coeffs[i+1][0], cmap='viridis')
        axes[i+1, 0].set_title(f'Horizontal Detail (Level {i+1})')
        
        # Vertical detail
        axes[i+1, 1].imshow(coeffs[i+1][1], cmap='viridis')
        axes[i+1, 1].set_title(f'Vertical Detail (Level {i+1})')
        
        # Diagonal detail
        axes[i+1, 2].imshow(coeffs[i+1][2], cmap='viridis')
        axes[i+1, 2].set_title(f'Diagonal Detail (Level {i+1})')
    
    plt.tight_layout()
    plt.suptitle('Wavelet Decomposition for Feature Extraction', fontsize=16, y=1.02)
    
    return fig

def visualize_lbp_features(image, radius=3, n_points=24):
    """
    Visualize Local Binary Pattern features
    
    Args:
        image: Input image
        radius: Radius for LBP calculation
        n_points: Number of points for LBP calculation
        
    Returns:
        Matplotlib figure
    """
    # Ensure image is 2D
    if len(image.shape) > 2:
        if image.shape[0] == 1:  # CHW format with 1 channel
            image = image[0]
        elif image.shape[2] == 1:  # HWC format with 1 channel
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # LBP image
    axes[1].imshow(lbp, cmap='viridis')
    axes[1].set_title('LBP Transformation')
    axes[1].axis('off')
    
    # Histogram
    axes[2].bar(range(len(hist)), hist)
    axes[2].set_title('LBP Histogram')
    axes[2].set_xlabel('LBP Uniform Pattern')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.suptitle('Local Binary Pattern (LBP) Feature Extraction', fontsize=16, y=1.02)
    
    return fig

def visualize_histogram_features(image, bins=256):
    """
    Visualize histogram features of an image
    
    Args:
        image: Input image
        bins: Number of bins for the histogram
        
    Returns:
        Matplotlib figure
    """
    # Ensure image is 2D
    if len(image.shape) > 2:
        if image.shape[0] == 1:  # CHW format with 1 channel
            image = image[0]
        elif image.shape[2] == 1:  # HWC format with 1 channel
            image = image[:, :, 0]
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            rgb_channels = True
    else:
        gray_image = image
        rgb_channels = False
    
    # Normalize if needed
    if gray_image.max() > 1.0:
        gray_image = gray_image / 255.0
        if rgb_channels:
            image = image / 255.0
    
    # Create figure
    if rgb_channels:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grayscale image
        axes[0, 1].imshow(gray_image, cmap='gray')
        axes[0, 1].set_title('Grayscale Image')
        axes[0, 1].axis('off')
        
        # RGB Histograms
        for i, color in enumerate(['red', 'green', 'blue']):
            axes[1, 0].hist(image[:, :, i].ravel(), bins=bins, alpha=0.7, color=color, label=color.capitalize())
        
        axes[1, 0].set_title('RGB Histograms')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Grayscale histogram
        axes[1, 1].hist(gray_image.ravel(), bins=bins, color='gray')
        axes[1, 1].set_title('Grayscale Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(gray_image, cmap='gray')
        axes[0].set_title('Grayscale Image')
        axes[0].axis('off')
        
        # Histogram
        axes[1].hist(gray_image.ravel(), bins=bins, color='gray')
        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Pixel Intensity')
        axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.suptitle('Histogram Features', fontsize=16, y=1.02)
    
    return fig

def visualize_model_predictions(images, true_labels, predictions, class_names=None, probabilities=None, n_samples=5):
    """
    Visualize model predictions
    
    Args:
        images: List or array of images
        true_labels: Ground truth labels
        predictions: Model predictions
        class_names: List of class names
        probabilities: Prediction probabilities (optional)
        n_samples: Number of samples to display
        
    Returns:
        Matplotlib figure
    """
    # If class names not provided, use numerical indices
    if class_names is None:
        classes = np.unique(np.concatenate([true_labels, predictions]))
        class_names = [f"Class {i}" for i in classes]
    
    # Select n_samples random indices
    n_images = len(images)
    indices = np.random.choice(n_images, min(n_samples, n_images), replace=False)
    
    # Extract samples
    sample_images = [images[i] for i in indices]
    sample_true = [true_labels[i] for i in indices]
    sample_pred = [predictions[i] for i in indices]
    
    # Create titles
    titles = []
    for i in range(len(indices)):
        true_class = class_names[sample_true[i]]
        pred_class = class_names[sample_pred[i]]
        
        if probabilities is not None:
            prob = probabilities[indices[i]][sample_pred[i]]
            title = f"True: {true_class}\nPred: {pred_class} ({prob:.2f})"
        else:
            title = f"True: {true_class}\nPred: {pred_class}"
        
        titles.append(title)
    
    # Plot images
    fig = plot_image_grid(sample_images, titles, cmaps='gray')
    plt.suptitle('Model Predictions', fontsize=16)
    
    return fig

def create_interactive_visualization(images, predictions, true_labels=None, class_names=None, features=None):
    """
    Create an interactive visualization using Plotly
    
    Args:
        images: List or array of images
        predictions: Model predictions
        true_labels: Ground truth labels (optional)
        class_names: List of class names
        features: Feature vectors for each image (optional)
        
    Returns:
        Plotly figure
    """
    # If class names not provided, use numerical indices
    if class_names is None:
        unique_classes = set(predictions)
        if true_labels is not None:
            unique_classes.update(true_labels)
        class_names = [f"Class {i}" for i in sorted(unique_classes)]
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=len(images), cols=2,
        specs=[[{"type": "image"}, {"type": "bar"}] for _ in range(len(images))],
        subplot_titles=[f"Image {i+1}" for i in range(len(images))] + 
                       [f"Prediction {i+1}" for i in range(len(images))]
    )
    
    # Add images and predictions
    for i, img in enumerate(images):
        # Normalize image
        if img.max() > 1.0:
            img = img / 255.0
        
        # Handle image dimensions
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        if len(img.shape) == 3 and img.shape[2] == 1:  # HWC with 1 channel
            img = img[:, :, 0]
        
        # Add image
        if len(img.shape) == 2:  # Grayscale
            fig.add_trace(
                go.Heatmap(z=img, showscale=False, colorscale='gray'),
                row=i+1, col=1
            )
        else:  # RGB
            fig.add_trace(
                go.Image(z=img),
                row=i+1, col=1
            )
        
        # Add predictions
        if isinstance(predictions[i], (int, np.integer)):
            # For class indices
            values = np.zeros(len(class_names))
            values[predictions[i]] = 1.0
        elif len(predictions[i]) == len(class_names):
            # For probabilities
            values = predictions[i]
        else:
            # Default case
            values = np.zeros(len(class_names))
            values[predictions[i]] = 1.0
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=class_names,
                y=values,
                marker_color=['#4e9af1' if j == predictions[i] else '#1e2130' for j in range(len(class_names))],
                text=[f"{v:.2f}" for v in values],
                textposition="auto"
            ),
            row=i+1, col=2
        )
        
        # Add true label indicator if available
        if true_labels is not None:
            fig.add_shape(
                type="rect",
                x0=true_labels[i]-0.4, x1=true_labels[i]+0.4,
                y0=0, y1=values[predictions[i]],
                line=dict(color="#00CC96", width=2),
                fillcolor="rgba(0,0,0,0)",
                row=i+1, col=2
            )
            fig.add_annotation(
                x=true_labels[i],
                y=0,
                text="True",
                showarrow=False,
                font=dict(color="#00CC96"),
                row=i+1, col=2
            )
    
    # Update layout
    fig.update_layout(
        height=300 * len(images),
        width=1000,
        showlegend=False,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1e2130',
        font=dict(color='white')
    )
    
    fig.update_xaxes(showticklabels=False, row=i+1, col=1)
    fig.update_yaxes(showticklabels=False, row=i+1, col=1)
    
    return fig

def create_feature_importance_heatmap(image, importance_map, title="Feature Importance"):
    """
    Create a heatmap overlay showing feature importance
    
    Args:
        image: Original image
        importance_map: Feature importance values (same shape as image)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Ensure same shape
    if image.shape[:2] != importance_map.shape[:2]:
        importance_map = cv2.resize(importance_map, (image.shape[1], image.shape[0]))
    
    # Normalize image if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    if len(image.shape) == 3 and image.shape[2] == 1:  # HWC with 1 channel
        image = image[:, :, 0]
    
    if len(image.shape) == 2:  # Grayscale
        axes[0].imshow(image, cmap='gray')
    else:  # RGB
        axes[0].imshow(image)
    
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Importance map
    im = axes[1].imshow(importance_map, cmap='hot', alpha=0.7)
    axes[1].set_title('Feature Importance')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    if len(image.shape) == 2:  # Grayscale
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(importance_map, cmap='hot', alpha=0.5)
    else:  # RGB
        axes[2].imshow(image)
        axes[2].imshow(importance_map, cmap='hot', alpha=0.5)
    
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_activation_maps(image, activation_maps, layer_names=None):
    """
    Visualize activation maps from a neural network
    
    Args:
        image: Original input image
        activation_maps: List of activation maps
        layer_names: Names of the layers (optional)
        
    Returns:
        Matplotlib figure
    """
    # If layer names not provided, generate generic names
    if layer_names is None:
        layer_names = [f"Layer {i+1}" for i in range(len(activation_maps))]
    
    # Normalize image if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Create figure
    n_maps = len(activation_maps)
    fig, axes = plt.subplots(n_maps + 1, 1, figsize=(8, 4 * (n_maps + 1)))
    
    # Original image
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    if len(image.shape) == 3 and image.shape[2] == 1:  # HWC with 1 channel
        image = image[:, :, 0]
    
    if len(image.shape) == 2:  # Grayscale
        axes[0].imshow(image, cmap='gray')
    else:  # RGB
        axes[0].imshow(image)
    
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Activation maps
    for i, (act_map, name) in enumerate(zip(activation_maps, layer_names)):
        # Average across channels for multichannel activations
        if len(act_map.shape) > 2:
            act_map = np.mean(act_map, axis=0)
        
        axes[i+1].imshow(act_map, cmap='viridis')
        axes[i+1].set_title(f'Activation Map: {name}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Neural Network Activation Maps', fontsize=16, y=1.02)
    
    return fig

def visualize_confusion_matrix(confusion_matrix, class_names=None, normalize=False):
    """
    Visualize confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize values
        
    Returns:
        Matplotlib figure
    """
    # If class names not provided, use numerical indices
    if class_names is None:
        class_names = [f"Class {i}" for i in range(confusion_matrix.shape[0])]
    
    # Normalize if requested
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    return plt.gcf()

def plot_training_history(history, metrics=['loss', 'accuracy']):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        # Get training and validation metrics
        train_metric = history.get(metric, [])
        val_metric = history.get(f'val_{metric}', [])
        
        epochs = range(1, len(train_metric) + 1)
        
        # Plot
        axes[i].plot(epochs, train_metric, 'b-', label=f'Training {metric}')
        if len(val_metric) > 0:
            axes[i].plot(epochs, val_metric, 'r-', label=f'Validation {metric}')
        
        axes[i].set_title(f'{metric.capitalize()} Over Epochs')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_signal_response(distance, signal_strength, threshold=None):
    """
    Plot microwave signal response with distance
    
    Args:
        distance: Array of distances
        signal_strength: Array of corresponding signal strengths
        threshold: Detection threshold (optional)
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(distance, signal_strength, 'b-', linewidth=2)
    
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label='Detection Threshold')
    
    plt.xlabel('Distance (mm)')
    plt.ylabel('Signal Strength')
    plt.title('Microwave Signal Response')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    return plt.gcf()

def create_gradcam_visualization(image, heatmap, alpha=0.4):
    """
    Create Grad-CAM visualization by overlaying heatmap on image
    
    Args:
        image: Original input image
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor for overlay
        
    Returns:
        Matplotlib figure
    """
    # Normalize image if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Handle image dimensions
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    
    if len(image.shape) == 3 and image.shape[2] == 1:  # HWC with 1 channel
        image = image[:, :, 0]
        is_grayscale = True
    elif len(image.shape) == 2:
        is_grayscale = True
    else:
        is_grayscale = False
    
    # Resize heatmap to match image size if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Create overlay
    if is_grayscale:
        # Convert grayscale to RGB for overlay
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=2)
        else:
            image_rgb = np.stack([image[:, :, 0]] * 3, axis=2)
        
        # Create overlay
        overlay = alpha * heatmap_colored + (1 - alpha) * image_rgb
    else:
        overlay = alpha * heatmap_colored + (1 - alpha) * image
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if is_grayscale and len(image.shape) == 2:
        axes[0].imshow(image, cmap='gray')
    elif is_grayscale:
        axes[0].imshow(image[:, :, 0], cmap='gray')
    else:
        axes[0].imshow(image)
        
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_colored)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Grad-CAM: Class Activation Mapping', fontsize=16, y=1.02)
    
    return fig

def create_streamlit_visualization(image, preprocessed_image, features, predictions, class_names=['Benign', 'Malignant']):
    """
    Create visualization elements for Streamlit UI
    
    Args:
        image: Original input image
        preprocessed_image: Preprocessed image
        features: Dictionary of extracted features
        predictions: Model predictions
        class_names: List of class names
        
    Returns:
        Dictionary of visualization elements for Streamlit
    """
    visualizations = {}
    
    # Image comparison
    visualizations['image_comparison'] = plot_image_grid(
        [image, preprocessed_image], 
        ['Original Image', 'Preprocessed Image'],
        cmaps='gray'
    )
    
    # Feature visualizations
    if 'wavelet' in features:
        visualizations['wavelet'] = visualize_wavelet_decomposition(preprocessed_image)
    
    if 'lbp' in features:
        visualizations['lbp'] = visualize_lbp_features(preprocessed_image)
    
    # Prediction visualization
    if len(predictions) == len(class_names):
        # For probability outputs
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class]
        
        # Create gauge chart for confidence
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': f"Prediction: {class_names[pred_class]}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4e9af1"},
                'steps': [
                    {'range': [0, 50], 'color': "#EF553B"},
                    {'range': [50, 75], 'color': "#FFA15A"},
                    {'range': [75, 100], 'color': "#00CC96"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            }
        ))
        
        gauge_fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#1e2130",
            font=dict(color="white")
        )
        
        visualizations['confidence_gauge'] = gauge_fig
        
        # Create bar chart for class probabilities
        bar_fig = go.Figure()
        
        bar_fig.add_trace(go.Bar(
            x=class_names,
            y=predictions,
            marker_color=['#4e9af1' if i == pred_class else '#1e2130' for i in range(len(class_names))],
            text=[f"{p*100:.1f}%" for p in predictions],
            textposition="auto"
        ))
        
        bar_fig.update_layout(
            title="Class Probabilities",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1e2130",
            font=dict(color="white"),
            yaxis=dict(range=[0, 1])
        )
        
        visualizations['probability_bar'] = bar_fig
    
    return visualizations

def create_microwave_imaging_visualization(image, signal_paths, signal_strengths):
    """
    Create visualization for microwave imaging simulation
    
    Args:
        image: Original mammogram image
        signal_paths: List of signal paths (list of coordinates)
        signal_strengths: Strength values for each path
        
    Returns:
        Matplotlib figure
    """
    # Normalize image if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Handle image dimensions
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    
    if len(image.shape) == 3 and image.shape[2] == 1:  # HWC with 1 channel
        image = image[:, :, 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    if len(image.shape) == 2:  # Grayscale
        ax.imshow(image, cmap='gray')
    else:  # RGB
        ax.imshow(image)
    
    # Normalize signal strengths for visualization
    max_strength = max(signal_strengths)
    norm_strengths = [s / max_strength for s in signal_strengths]
    
    # Plot signal paths with colors based on strength
    for path, strength in zip(signal_paths, norm_strengths):
        # Convert strength to color (red to green)
        color = (1 - strength, strength, 0)
        
        # Extract coordinates
        x_coords, y_coords = zip(*path)
        
        # Plot path
        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, max_strength))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Signal Strength')
    
    ax.set_title('Microwave Signal Paths')
    ax.axis('off')
    
    return fig

def visualize_signal_propagation(original_image, regions_of_interest, signal_strengths):
    """
    Visualize microwave signal propagation through tissue
    
    Args:
        original_image: Original mammogram image
        regions_of_interest: List of ROI coordinates (x, y, width, height)
        signal_strengths: Signal strength values for corresponding regions
        
    Returns:
        Matplotlib figure
    """
    # Normalize image if needed
    if original_image.max() > 1.0:
        original_image = original_image / 255.0
    
    # Handle image dimensions
    if len(original_image.shape) == 3 and original_image.shape[0] in [1, 3]:  # CHW format
        original_image = np.transpose(original_image, (1, 2, 0))
    
    if len(original_image.shape) == 3 and original_image.shape[2] == 1:  # HWC with 1 channel
        original_image = original_image[:, :, 0]
    
    # Create signal propagation visualization
    height, width = original_image.shape[:2]
    signal_map = np.zeros((height, width))
    
    # Fill signal map
    for (x, y, w, h), strength in zip(regions_of_interest, signal_strengths):
        signal_map[y:y+h, x:x+w] = strength
    
    # Apply Gaussian blur to simulate signal propagation
    signal_map = cv2.GaussianBlur(signal_map, (15, 15), 0)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    if len(original_image.shape) == 2:  # Grayscale
        axes[0].imshow(original_image, cmap='gray')
    else:  # RGB
        axes[0].imshow(original_image)
    
    axes[0].set_title('Original Mammogram')
    axes[0].axis('off')
    
    # Signal map
    im = axes[1].imshow(signal_map, cmap='plasma')
    axes[1].set_title('Microwave Signal Propagation')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    if len(original_image.shape) == 2:  # Grayscale
        axes[2].imshow(original_image, cmap='gray')
        axes[2].imshow(signal_map, cmap='plasma', alpha=0.6)
    else:  # RGB
        axes[2].imshow(original_image)
        axes[2].imshow(signal_map, cmap='plasma', alpha=0.6)
    
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle('Microwave Signal Propagation Through Breast Tissue', fontsize=16)
    plt.tight_layout()
    
    return fig

def create_detailed_analysis_report(image, features, predictions, true_label=None):
    """
    Create a detailed visual analysis report
    
    Args:
        image: Original mammogram image
        features: Dictionary of extracted features
        predictions: Model predictions (probabilities)
        true_label: Ground truth label (optional)
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    if len(image.shape) == 3 and image.shape[2] == 1:  # HWC with 1 channel
        image = image[:, :, 0]
    
    if len(image.shape) == 2:  # Grayscale
        ax1.imshow(image, cmap='gray')
    else:  # RGB
        ax1.imshow(image)
    
    ax1.set_title('Original Mammogram')
    ax1.axis('off')
    
    # Extracted features visualization
    if 'wavelet' in features:
        # Simplified wavelet visualization
        ax2 = fig.add_subplot(gs[0, 1])
        wavelet_features = features['wavelet'].reshape(-1, 4)
        ax2.bar(range(len(wavelet_features)), wavelet_features.flatten())
        ax2.set_title('Wavelet Features')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Value')
    
    if 'lbp' in features:
        # LBP histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(range(len(features['lbp'])), features['lbp'])
        ax3.set_title('LBP Features')
        ax3.set_xlabel('LBP Pattern')
        ax3.set_ylabel('Frequency')
    
    # Prediction visualization
    ax4 = fig.add_subplot(gs[1, :])
    
    pred_class = np.argmax(predictions)
    class_names = ['Benign', 'Malignant']
    colors = ['#4e9af1', '#EF553B']
    
    ax4.bar(class_names, predictions, color=colors)
    ax4.set_ylim([0, 1])
    ax4.set_title('Classification Probabilities')
    ax4.set_ylabel('Probability')
    
    # Add percentage labels
    for i, p in enumerate(predictions):
        ax4.text(i, p + 0.02, f"{p*100:.1f}%", ha='center')
    
    # Add true label marker if available
    if true_label is not None:
        ax4.scatter(class_names[true_label], predictions[true_label], marker='*', 
                   s=200, color='yellow', edgecolor='black', zorder=10)
        ax4.text(class_names[true_label], predictions[true_label] - 0.08, 
                'True Label', ha='center', fontweight='bold')
    
    # Confidence gauge
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create a simple gauge
    confidence = predictions[pred_class]
    theta = np.linspace(0, 180, 100)
    r = 1
    
    # Convert to cartesian
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    
    # Plot the gauge background
    ax5.plot(x, y, color='lightgray', linewidth=20)
    
    # Plot the value
    value_theta = 180 * (1 - confidence)
    value_idx = int((value_theta / 180) * len(theta))
    ax5.plot(x[:value_idx], y[:value_idx], color=colors[pred_class], linewidth=20)
    
    # Add gauge ticks
    for i in range(0, 101, 20):
        tick_theta = 180 * (1 - i/100)
        tick_x = r * np.cos(np.radians(tick_theta))
        tick_y = r * np.sin(np.radians(tick_theta))
        ax5.plot([0, tick_x*1.1], [0, tick_y*1.1], color='black', linewidth=1)
        ax5.text(tick_x*1.2, tick_y*1.2, f"{i}%", ha='center', va='center')
    
    # Add needle
    ax5.plot([0, x[value_idx]], [0, y[value_idx]], color='black', linewidth=2)
    
    # Add confidence text
    ax5.text(0, -0.5, f"Confidence: {confidence*100:.1f}%", ha='center', fontsize=14, fontweight='bold')
    
    # Prediction text
    ax5.text(0, -0.7, f"Prediction: {class_names[pred_class]}", ha='center', fontsize=16, 
             color=colors[pred_class], fontweight='bold')
    
    ax5.set_xlim([-1.5, 1.5])
    ax5.set_ylim([-1, 1.2])
    ax5.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Detailed Analysis Report', fontsize=16, y=0.98)
    
    return fig

if __name__ == "__main__":
    print("This module provides visualization functions for medical image classification")
    print("Import and use these functions in your main application")