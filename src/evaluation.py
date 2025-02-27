import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def evaluate_model(model, test_loader, device='cpu', threshold=0.5):
    """
    Evaluate the model on test data
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader containing test data
        device: Device to run evaluation on ('cpu' or 'cuda')
        threshold: Classification threshold for binary classification
        
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    model.eval()
    model.to(device)
    
    y_true = []
    y_pred = []
    y_scores = []
    all_features = []
    all_images = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # For binary classification
            if outputs.shape[1] == 2:
                scores = torch.softmax(outputs, dim=1)
                preds = (scores[:, 1] > threshold).long()
                scores = scores[:, 1].cpu().numpy()  # Get probability of positive class
            # For multi-class
            else:
                scores = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1)
                
            # Collect results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(scores)
            
            # Store some examples for visualization
            if len(all_images) < 20:  # Store just a few images for visualization
                all_images.extend(inputs.cpu().numpy()[:5])
            
    # Calculate metrics
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred, average='weighted')
    results['recall'] = recall_score(y_true, y_pred, average='weighted')
    results['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        results['roc_auc'] = auc(fpr, tpr)
        results['fpr'] = fpr
        results['tpr'] = tpr
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Store predictions and ground truth
    results['y_true'] = y_true
    results['y_pred'] = y_pred
    results['y_scores'] = y_scores
    results['images'] = all_images
    
    return results

def calculate_class_metrics(results, class_names=None):
    """
    Calculate per-class metrics
    
    Args:
        results: Results dictionary from evaluate_model
        class_names: List of class names for display
        
    Returns:
        DataFrame with per-class metrics
    """
    cm = results['confusion_matrix']
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # If class names not provided, use numerical indices
    if class_names is None:
        classes = np.unique(y_true)
        class_names = [f"Class {i}" for i in classes]
    
    # Calculate per-class metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    
    return metrics_df

def plot_roc_curve(results, title='Receiver Operating Characteristic'):
    """
    Plot ROC curve for binary classification
    
    Args:
        results: Results dictionary from evaluate_model
        title: Title for the plot
    """
    if 'roc_auc' not in results:
        print("ROC curve is only available for binary classification")
        return
    
    plt.figure(figsize=(10, 8))
    plt.plot(results['fpr'], results['tpr'], 
             color='darkorange', lw=2, 
             label=f'ROC curve (area = {results["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    return plt.gcf()

def plot_confusion_matrix(results, class_names=None, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        results: Results dictionary from evaluate_model
        class_names: List of class names for display
        title: Title for the plot
    """
    cm = results['confusion_matrix']
    
    # If class names not provided, use numerical indices
    if class_names is None:
        classes = np.unique(results['y_true'])
        class_names = [f"Class {i}" for i in classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    return plt.gcf()

def analyze_misclassifications(results, class_names=None, num_examples=5):
    """
    Analyze and visualize misclassified examples
    
    Args:
        results: Results dictionary from evaluate_model
        class_names: List of class names for display
        num_examples: Number of misclassified examples to show
        
    Returns:
        DataFrame with misclassification statistics
    """
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # If class names not provided, use numerical indices
    if class_names is None:
        classes = np.unique(y_true)
        class_names = [f"Class {i}" for i in classes]
    
    # Find misclassifications
    misclassified = np.where(np.array(y_true) != np.array(y_pred))[0]
    
    # Count misclassifications per class
    misclass_counts = {}
    for true_cls in range(len(class_names)):
        misclass_counts[true_cls] = {}
        for pred_cls in range(len(class_names)):
            if true_cls != pred_cls:
                count = np.sum((np.array(y_true) == true_cls) & (np.array(y_pred) == pred_cls))
                if count > 0:
                    misclass_counts[true_cls][pred_cls] = count
    
    # Create summary DataFrame
    misclass_data = []
    for true_cls, pred_dict in misclass_counts.items():
        for pred_cls, count in pred_dict.items():
            misclass_data.append({
                'True Class': class_names[true_cls],
                'Predicted Class': class_names[pred_cls],
                'Count': count
            })
    
    misclass_df = pd.DataFrame(misclass_data)
    misclass_df = misclass_df.sort_values('Count', ascending=False)
    
    return misclass_df

def generate_performance_report(results, class_names=None, output_file=None):
    """
    Generate a comprehensive performance report
    
    Args:
        results: Results dictionary from evaluate_model
        class_names: List of class names
        output_file: Path to save report (if None, print to console)
    """
    # Overall metrics
    overall = (
        f"Model Performance Summary\n"
        f"========================\n"
        f"Accuracy:  {results['accuracy']:.4f}\n"
        f"Precision: {results['precision']:.4f}\n"
        f"Recall:    {results['recall']:.4f}\n"
        f"F1 Score:  {results['f1']:.4f}\n"
    )
    
    # Add ROC AUC for binary classification
    if 'roc_auc' in results:
        overall += f"ROC AUC:   {results['roc_auc']:.4f}\n"
    
    # Per-class metrics
    class_metrics = calculate_class_metrics(results, class_names)
    
    # Misclassification analysis
    misclass_df = analyze_misclassifications(results, class_names)
    
    # Combine report
    report = (
        overall + 
        "\nPer-Class Metrics\n"
        "=================\n" + 
        class_metrics.to_string() + 
        "\n\nMisclassification Analysis\n"
        "=========================\n" + 
        misclass_df.to_string() + 
        "\n"
    )
    
    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
    else:
        print(report)
    
    return report

def evaluate_feature_importance(model, test_loader, device='cpu'):
    """
    Evaluate the feature importance using occlusion sensitivity
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader containing test data
        device: Device to run evaluation on ('cpu' or 'cuda')
        
    Returns:
        Heatmap of feature importance
    """
    # This is a simplified implementation of occlusion sensitivity
    # For a real implementation, you'd need to occlude different regions of the image
    # and measure the change in prediction
    print("Feature importance analysis requires model-specific implementation")
    print("This is a placeholder function")
    
    return None

if __name__ == "__main__":
    print("This module provides evaluation functions for medical image classification models")
    print("Import and use these functions in your main script")