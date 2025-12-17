# Parkinson's Disease Detection System

A machine learning system for detecting Parkinson's Disease using handwriting signals data. This project implements an Attention-Based Bidirectional LSTM model with comprehensive data preprocessing, feature engineering, and a user-friendly GUI application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the GUI Application](#running-the-gui-application)
  - [Running Jupyter Notebooks](#running-jupyter-notebooks)
- [Project Structure](#project-structure)
- [Model Variants](#model-variants)
- [Requirements](#requirements)
- [Data Format](#data-format)
- [Citation](#citation)

## 🎯 Overview

This thesis project focuses on developing a deep learning-based system for Parkinson's Disease detection using handwriting dynamics features. The system analyzes five key handwriting dynamics channels:

- Fingergrip
- Axial Pressure
- Tilt X
- Tilt Y
- Tilt Z

## ✨ Features

- **Attention-Based Bi-LSTM Model**: Deep learning architecture with attention mechanism for temporal pattern recognition
- **Data Augmentation**: Advanced augmentation techniques to improve model robustness
- **Feature Engineering**: Statistical and temporal feature extraction
- **Interactive GUI**: Complete pipeline with preprocessing, training, and evaluation
- **Comprehensive Evaluation**: Multi-level evaluation (sequence, file, and person levels)
- **Attention Visualization**: Understanding model decision-making through attention weight analysis
- **Cross-Validation**: K-fold cross-validation for robust model evaluation

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/stiipp/DATASCI_17_Thesis.git
cd DATASCI_17_Thesis
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm tensorflow pillow
```

Or install all packages individually:

```bash
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0
pip install scipy>=1.7.0
pip install scikit-learn>=0.24.0
pip install tqdm>=4.62.0
pip install tensorflow>=2.10.0
pip install pillow>=8.3.0
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
```

## 📖 Usage

### Running the GUI Application

The GUI application provides a complete pipeline for Parkinson's Disease detection with an intuitive interface.

1. **Navigate to the GUI directory:**
   ```bash
   cd GUI
   ```

2. **Run the application:**
   ```bash
   python parkinson_detection_gui_complete.py
   ```

3. **Using the GUI:**
   - **Tab 1 - Data Loading**: 
     - Load healthy control data folder
     - Load patient data folder
     - Split data into training and testing sets
   
   - **Tab 2 - Preprocessing & Training**:
     - Configure preprocessing parameters (sequence length, channels, etc.)
     - Apply data augmentation
     - Perform feature engineering
     - Train the Attention-Based Bi-LSTM model
     - Monitor training progress in real-time
   
   - **Tab 3 - Evaluation**:
     - Evaluate model on test set
     - View performance metrics (accuracy, precision, recall, F1-score)
     - Visualize confusion matrix
     - Analyze ROC curves
     - Export results
   
   - **Tab 4 - Attention Visualization**:
     - Select individual sequences
     - Visualize attention weights
     - Understand model decision-making

### Running Jupyter Notebooks

The project includes multiple Jupyter notebooks for different experimental configurations.

1. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Navigate to the desired pipeline:**
   - `Pipelines (Updated)/Pipelines/With Aug and FE/` - Full pipeline with augmentation and feature engineering
   - `Pipelines (Updated)/Pipelines/With Aug and without FE/` - Augmentation only
   - `Pipelines (Updated)/Pipelines/Without Aug and with FE/` - Feature engineering only
   - `Pipelines (Updated)/Pipelines/Without Aug and FE/` - Baseline model
   - `Pipelines (Updated)/Pipelines/Without Attention/` - Bi-LSTM without attention mechanism

4. **Available Notebooks:**
   - `Data Pre-Processing.ipynb` - Data loading and preprocessing
   - `Cross Validation for Attention-Based Bi-LSTM.ipynb` - K-fold cross-validation
   - `Final Training for Attention-Based Bi-LSTM.ipynb` - Final model training
   - `Multi-Seed Training for Attention-Based Bi-LSTM.ipynb` - Multiple seed experiments
   - `[With Statistical Tests] Final Training.ipynb` - Training with statistical analysis

## 📁 Project Structure

```
DATASCI_17_Thesis/
│
├── GUI/
│   └── parkinson_detection_gui_complete.py    # Complete GUI application
│
└── Pipelines (Updated)/
    └── Pipelines/
        ├── With Aug and FE/                    # Full pipeline
        │   ├── Data Pre-Processing.ipynb
        │   ├── Cross Validation for Attention-Based Bi-LSTM with Aug and FE.ipynb
        │   ├── Final Training for Attention-Based Bi-LSTM with Aug and FE.ipynb
        │   ├── Multi-Seed Training for Attention-Based Bi-LSTM with Aug and FE.ipynb
        │   └── [With Statistical Tests] Final Training for Attention-Based Bi-LSTM with Aug and FE.ipynb
        │
        ├── With Aug and without FE/            # Augmentation only
        ├── Without Aug and with FE/            # Feature engineering only
        ├── Without Aug and FE/                 # Baseline
        └── Without Attention/                  # Bi-LSTM without attention
```

## 🔬 Model Variants

The project explores multiple model configurations:

1. **Attention-Based Bi-LSTM with Aug and FE** (Recommended)
   - Full pipeline with data augmentation and feature engineering
   - Best performance across all metrics

2. **Attention-Based Bi-LSTM with Aug and without FE**
   - Data augmentation without statistical features

3. **Attention-Based Bi-LSTM without Aug and with FE**
   - Feature engineering without augmentation

4. **Attention-Based Bi-LSTM without Aug and FE**
   - Baseline attention model

5. **Bi-LSTM without Attention (with Aug and FE)**
   - Standard Bi-LSTM for comparison

## 📦 Requirements

### Core Dependencies

- **Python**: 3.8+
- **TensorFlow**: 2.10.0+ (Deep learning framework)
- **NumPy**: 1.21.0+ (Numerical computing)
- **Pandas**: 1.3.0+ (Data manipulation)
- **Scikit-learn**: 0.24.0+ (Machine learning utilities)

### Visualization

- **Matplotlib**: 3.4.0+ (Plotting)
- **Seaborn**: 0.11.0+ (Statistical visualization)

### Data Processing

- **SciPy**: 1.7.0+ (Scientific computing)
- **tqdm**: 4.62.0+ (Progress bars)

### GUI

- **tkinter**: Included with Python (GUI framework)
- **Pillow**: 8.3.0+ (Image processing)

### Optional (for GPU acceleration)

- **CUDA Toolkit**: 11.2+ (NVIDIA GPU support)
- **cuDNN**: 8.1+ (GPU-accelerated deep learning)

## 📊 Data Format

The system expects handwriting dynamics data in the following format:

- **Folder Structure**: Separate folders for healthy controls and patients
- **File Format**: CSV or compatible format with time-series data
- **Required Channels**: 
  - Fingergrip
  - Axial_Pressure
  - Tilt_X
  - Tilt_Y
  - Tilt_Z
- **Data Organization**: Each file represents one handwriting sample

## 📝 Notes

- The model uses a fixed random seed (42) for reproducibility across all experiments
- Training requires significant computational resources; GPU acceleration is recommended
- The GUI application provides real-time logging and progress updates during training
- All models include comprehensive evaluation at sequence, file, and person levels
- Attention visualizations help interpret model predictions

## 🤝 Contributing

This is a thesis project. For questions or collaboration inquiries, please open an issue on GitHub.

## 📄 License

This project is part of academic research. Please contact the author for usage rights and licensing information.

---

**Last Updated**: December 2025
