# AI-Driven Predictive Maintenance for Industrial Equipment

This repository implements an AI-powered predictive maintenance solution for industrial machinery using machine learning algorithms to forecast equipment failures and enable proactive maintenance actions.

## 📋 Overview

Based on the research paper "AI-Driven Predictive Maintenance for Industrial Equipment: Enhancing Operational Efficiency through Machine Learning" by Kalita & Das, this system demonstrates how to:

- Predict equipment failures using sensor data
- Minimize unplanned downtime
- Reduce maintenance costs
- Improve operational efficiency

## 🔧 Features

- **Multiple ML Models**: Random Forest, SVM, Logistic Regression, and K-Nearest Neighbors
- **Data Preprocessing**: Handles missing values, feature scaling, and class imbalance
- **Real-time Monitoring**: Simulates live equipment monitoring
- **High Accuracy**: Achieves 98%+ accuracy with Random Forest and SVM models
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score metrics
- **Model Persistence**: Save and load trained models

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/predictive-maintenance-ai.git
cd predictive-maintenance-ai
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage

```python
from predictive_maintenance import PredictiveMaintenanceSystem

# Initialize the system
pm_system = PredictiveMaintenanceSystem()

# Load data (synthetic data for demonstration)
data, target = pm_system.load_data(synthetic=True)

# Preprocess data
X_train, X_test, y_train, y_test = pm_system.preprocess_data()

# Train models
pm_system.initialize_models()
pm_system.train_models(X_train, y_train)

# Evaluate models
results = pm_system.evaluate_models(X_test, y_test)
pm_system.display_results()

# Run real-time monitoring simulation
pm_system.real_time_monitoring_simulation(X_test, y_test)
```

#### Using Your Own Data

```python
# Load your SECOM dataset
data, target = pm_system.load_data(data_path='path/to/your/secom_dataset.csv')
```

#### Single Prediction

```python
# Predict failure for a single sample
sample_data = [...]  # Your sensor readings
prediction, probability = pm_system.predict_single_sample(sample_data, 'Random Forest')
```

## 📊 Performance Results

Based on the SECOM dataset, the models achieve the following performance:

| Model | Accuracy (%) | Precision | Recall | F1 Score | F-Measure |
|-------|-------------|-----------|--------|----------|-----------|
| **Random Forest** | 98.81 | 0.99 | 0.99 | 0.99 | 0.989 |
| **SVM** | 98.98 | 0.98 | 1.00 | 0.99 | 0.990 |
| **Logistic Regression** | 91.30 | 0.85 | 0.85 | 0.92 | 0.850 |
| **KNN** | 59.73 | 0.54 | 1.00 | 0.70 | 0.701 |

## 📁 Project Structure

```
predictive-maintenance-ai/
├── predictive_maintenance.py    # Main system implementation
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── demo.ipynb                 # Jupyter notebook demo
├── models/                    # Saved models directory
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   └── imputer.pkl
├── data/                      # Data directory
│   └── secom_dataset.csv     # SECOM dataset (if available)
└── examples/                  # Example scripts
    └── basic_usage.py
```

## 🔄 System Architecture

The system follows a comprehensive pipeline:

1. **Data Collection**: Industrial sensor data acquisition
2. **Data Preprocessing**: 
   - Missing value imputation
   - Feature scaling
   - Class imbalance handling with SMOTE
3. **Model Training**: Multiple ML algorithms
4. **Real-time Monitoring**: Continuous prediction and alerting
5. **Maintenance Actions**: Proactive maintenance scheduling

## 📈 Key Components

### Data Preprocessing
- **Missing Value Handling**: Mean imputation for sensor readings
- **Feature Scaling**: StandardScaler for normalization
- **Class Imbalance**: SMOTE for minority class oversampling

### Machine Learning Models
- **Random Forest**: Ensemble method with 100 trees
- **Support Vector Machine**: Linear kernel with balanced classes
- **Logistic Regression**: Simple yet effective linear model
- **K-Nearest Neighbors**: Distance-based classification

### Real-time Monitoring
- Continuous sensor data processing
- Instant failure prediction
- Automated alert system
- Maintenance scheduling integration

## 🛠️ Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn
- joblib

## 📝 Usage Examples

### Running the Demo

```bash
python predictive_maintenance.py
```

### Custom Model Training

```python
# Initialize with custom parameters
pm_system = PredictiveMaintenanceSystem()

# Load and preprocess your data
data, target = pm_system.load_data(data_path='your_data.csv')
X_train, X_test, y_train, y_test = pm_system.preprocess_data()

# Train specific models
pm_system.initialize_models()
pm_system.train_models(X_train, y_train)

# Evaluate and save
results = pm_system.evaluate_models(X_test, y_test)
pm_system.save_models('./your_models')
```

## 🎯 Future Enhancements

- **Deep Learning Models**: LSTM networks for temporal dependencies
- **Edge Computing**: Model deployment on edge devices
- **Digital Twin Integration**: Virtual equipment modeling
- **Explainable AI**: Model interpretability features
- **Multi-sensor Fusion**: Enhanced sensor data integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Support

If you have any questions or need help with the implementation, please:

1. Check the documentation
2. Open an issue on GitHub
3. Contact the maintainers

---

**Note**: This system is designed for educational and research purposes. For production deployment, ensure proper validation, testing, and compliance with industrial safety standards.
