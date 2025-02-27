# MediScan: Advanced Breast Cancer Detection System

## Overview

![Image](https://github.com/user-attachments/assets/6276974f-b38e-47fe-835b-8f3543bbb871)
![Image](https://github.com/user-attachments/assets/671f4219-d8ce-4368-8734-22bba6be718a)
![Image](https://github.com/user-attachments/assets/1e3931b7-74df-4b9c-98a5-a43498c1a34a)

MediScan is a state-of-the-art medical imaging application that leverages microwave imaging technology and deep learning to detect breast cancer. The system processes mammogram images through a sophisticated pipeline of preprocessing, feature extraction, and classification to provide reliable diagnostic assessments with high accuracy.

## Key Features

- **Advanced Microwave Imaging Simulation**: Simulates microwave signal propagation through breast tissue for enhanced detection capabilities
- **Multi-stage Image Processing Pipeline**: Noise reduction, contrast enhancement, and region-of-interest segmentation
- **Wavelet-based Feature Extraction**: Utilizes wavelet transforms particularly relevant to microwave imaging analysis
- **Local Binary Pattern (LBP) Analysis**: Extracts texture features essential for differentiating tissue types
- **Deep Learning Classification**: PyTorch-based convolutional neural network for binary classification (Benign/Malignant)
- **Interactive Visualization**: Comprehensive visualization suite for model interpretability and feature analysis
- **Intuitive Web Interface**: Streamlit-powered user interface with modern design focused on clinical usability
- **Explainable AI**: Feature importance analysis and Grad-CAM visualization to explain model decisions

## Project Structure

```
medical_image_classifier/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
│
├── data/                   # Data directory
│   ├── raw/                # Raw image data
│   └── processed/          # Processed image data
│
├── models/                 # Model storage
│   ├── model.py            # Model architecture definition
│   ├── train.py            # Training script
│   └── saved_models/       # Saved model weights
│
└── src/                    # Source code
    ├── preprocessing.py    # Image preprocessing utilities
    ├── feature_extraction.py # Signal processing for feature extraction
    ├── evaluation.py       # Model evaluation utilities
    └── visualization.py    # Visualization utilities
```

## Pipeline Operation

The MediScan pipeline consists of the following stages:

1. **Image Preprocessing**
   - Noise reduction using Gaussian blur
   - Contrast enhancement with CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Breast region segmentation
   - Normalization of pixel values

2. **Feature Extraction**
   - Wavelet transform to extract frequency-based features (db1 wavelet at level 3)
   - Local Binary Pattern analysis for texture feature extraction
   - Statistical feature generation (means, standard deviations, etc.)

3. **Model Inference**
   - Input preprocessing and batching
   - Forward pass through the CNN model
   - Probability calculation with softmax
   - Classification (Benign/Malignant)

4. **Results Visualization**
   - Feature importance display
   - Confidence metrics visualization
   - Interactive analysis of model decisions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-image-classifier.git
cd medical-image-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Application

To launch the web interface:

```bash
streamlit run app.py
```

This will start the application on `localhost:8501` by default.

### Training a New Model

To train a new model on your dataset:

```bash
python models/train.py --data_dir path/to/dataset --epochs 50 --batch_size 32
```

### Performing Evaluations

To evaluate a trained model:

```bash
python src/evaluation.py --model_path models/saved_models/best_model.pth --test_data path/to/test_data
```

## Deployment

### Local Deployment

For local deployment, simply run the Streamlit application:

```bash
streamlit run app.py
```

### Cloud Deployment (AWS)

1. **Setup EC2 Instance**:
   - Launch an EC2 instance with Ubuntu (t2.large or better recommended)
   - Configure security groups to allow HTTP/HTTPS traffic

2. **Environment Setup**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip python3-dev
   git clone https://github.com/yourusername/medical-image-classifier.git
   cd medical-image-classifier
   pip3 install -r requirements.txt
   ```

3. **Running as a Service**:
   - Create a systemd service file:
   ```bash
   sudo nano /etc/systemd/system/mediscan.service
   ```
   
   - Add the following content:
   ```
   [Unit]
   Description=MediScan Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/medical-image-classifier
   ExecStart=/home/ubuntu/medical-image-classifier/venv/bin/streamlit run app.py --server.port=80
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
   
   - Enable and start the service:
   ```bash
   sudo systemctl enable mediscan.service
   sudo systemctl start mediscan.service
   ```

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t mediscan:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 mediscan:latest
   ```

### Kubernetes Deployment

1. **Create a Kubernetes deployment YAML**:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: mediscan
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: mediscan
     template:
       metadata:
         labels:
           app: mediscan
       spec:
         containers:
         - name: mediscan
           image: mediscan:latest
           ports:
           - containerPort: 8501
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: mediscan-service
   spec:
     selector:
       app: mediscan
     ports:
     - port: 80
       targetPort: 8501
     type: LoadBalancer
   ```

2. **Apply the deployment**:
   ```bash
   kubectl apply -f deployment.yaml
   ```

## Performance

The current model achieves:
- **Accuracy**: 92% on the test dataset
- **Sensitivity**: 89% (ability to detect malignant cases)
- **Specificity**: 94% (ability to correctly identify benign cases)
- **Processing Time**: ~1.2 seconds per image

## Future Work

- Integration with real-time microwave imaging hardware
- Multi-class classification for different types of breast abnormalities
- Ensemble models for improved accuracy
- Mobile application for remote diagnostics
- Integration with hospital PACS (Picture Archiving and Communication System)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Mini-MIAS and CBIS-DDSM datasets for providing training data
- PyTorch team for the deep learning framework
- Streamlit for the web application framework

## Contact

For any questions or feedback, please contact: mohammad.sameer@epita.fr
