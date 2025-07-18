"""
Utility functions for AI-Driven Predictive Maintenance System

This module contains helper functions for data processing, visualization,
logging, and other utility operations.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Import configuration
from config import LOGGING_CONFIG, VISUALIZATION_CONFIG, STORAGE_CONFIG

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(log_level: str = None, log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration for the predictive maintenance system
    
    Parameters:
    log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file (str): Path to log file
    
    Returns:
    logging.Logger: Configured logger instance
    """
    if log_level is None:
        log_level = LOGGING_CONFIG['level']
    if log_file is None:
        log_file = LOGGING_CONFIG['log_file']
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('PredictiveMaintenance')
    logger.info("Logging system initialized")
    
    return logger

def log_model_performance(logger: logging.Logger, model_name: str, metrics: Dict[str, float]) -> None:
    """Log model performance metrics"""
    logger.info(f"Model Performance - {model_name}:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

# =============================================================================
# DATA UTILITIES
# =============================================================================

def validate_data(data: pd.DataFrame, target: pd.Series) -> bool:
    """
    Validate input data for the predictive maintenance system
    
    Parameters:
    data (pd.DataFrame): Feature data
    target (pd.Series): Target labels
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    try:
        # Check if data is not empty
        if data.empty or target.empty:
            raise ValueError("Data or target is empty")
        
        # Check if dimensions match
        if len(data) != len(target):
            raise ValueError("Data and target have different lengths")
        
        # Check for valid target values
        unique_targets = target.unique()
        if not all(val in [-1, 1] for val in unique_targets):
            raise ValueError("Target values must be -1 (normal) or 1 (failure)")
        
        # Check for reasonable data ranges
        if data.isnull().all().any():
            warnings.warn("Some features have all missing values")
        
        return True
        
    except Exception as e:
        warnings.warn(f"Data validation failed: {e}")
        return False

def generate_synthetic_secom_data(n_samples: int = 1567, n_features: int = 590, 
                                missing_rate: float = 0.05, failure_rate: float = 0.07,
                                random_seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic data similar to SECOM dataset
    
    Parameters:
    n_samples (int): Number of samples
    n_features (int): Number of features
    missing_rate (float): Proportion of missing values
    failure_rate (float): Proportion of failure samples
    random_seed (int): Random seed for reproducibility
    
    Returns:
    Tuple[pd.DataFrame, pd.Series]: Features and target data
    """
    np.random.seed(random_seed)
    
    # Generate correlated sensor data
    base_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features) + 0.1 * np.random.random((n_features, n_features)),
        size=n_samples
    )
    
    # Add some non-linear relationships
    for i in range(0, n_features, 10):
        if i + 1 < n_features:
            base_data[:, i] = base_data[:, i] * np.sin(base_data[:, i + 1])
    
    # Create target variable with some dependency on features
    feature_sum = np.sum(base_data[:, :10], axis=1)
    failure_prob = 1 / (1 + np.exp(-(feature_sum - np.percentile(feature_sum, 93))))
    target = np.random.binomial(1, failure_prob)
    target = np.where(target == 0, -1, 1)  # Convert to -1/1 format
    
    # Ensure correct failure rate
    n_failures = int(n_samples * failure_rate)
    failure_indices = np.random.choice(n_samples, n_failures, replace=False)
    target[:] = -1  # Set all to normal
    target[failure_indices] = 1  # Set selected to failure
    
    # Add missing values
    missing_mask = np.random.random((n_samples, n_features)) < missing_rate
    base_data[missing_mask] = np.nan
    
    # Create DataFrames
    feature_names = [f'sensor_{i}' for i in range(n_features)]
    data = pd.DataFrame(base_data, columns=feature_names)
    target = pd.Series(target, name='target')
    
    return data, target

def calculate_missing_value_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive missing value statistics"""
    total_cells = data.size
    missing_cells = data.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    features_with_missing = data.isnull().any()
    n_features_with_missing = features_with_missing.sum()
    
    samples_with_missing = data.isnull().any(axis=1)
    n_samples_with_missing = samples_with_missing.sum()
    
    return {
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'missing_percentage': missing_percentage,
        'features_with_missing': n_features_with_missing,
        'samples_with_missing': n_samples_with_missing,
        'missing_per_feature': data.isnull().sum().to_dict(),
        'missing_per_sample': data.isnull().sum(axis=1).describe().to_dict()
    }

# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use(VISUALIZATION_CONFIG['style'])
    sns.set_palette(VISUALIZATION_CONFIG['color_palette'])
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figure_size']
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']

def plot_confusion_matrix_enhanced(y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str, save_path: str = None) -> None:
    """
    Plot enhanced confusion matrix with additional statistics
    
    Parameters:
    y_true (np.ndarray): True labels
    y_pred (np.ndarray): Predicted labels
    model_name (str): Name of the model
    save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add statistics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    stats_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.text(2.5, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], 
                   format=VISUALIZATION_CONFIG['plot_format'])
    
    plt.show()

def plot_feature_importance(feature_importance: np.ndarray, feature_names: List[str],
                          model_name: str, top_n: int = 20, save_path: str = None) -> None:
    """
    Plot feature importance for tree-based models
    
    Parameters:
    feature_importance (np.ndarray): Feature importance values
    feature_names (List[str]): Names of features
    model_name (str): Name of the model
    top_n (int): Number of top features to display
    save_path (str): Path to save the plot
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(importance_df['importance']):
        plt.text(v + max(importance_df['importance'])*0.01, i, f'{v:.4f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], 
                   format=VISUALIZATION_CONFIG['plot_format'])
    
    plt.show()

def plot_class_distribution(target: pd.Series, title: str = "Class Distribution",
                          save_path: str = None) -> None:
    """
    Plot class distribution
    
    Parameters:
    target (pd.Series): Target variable
    title (str): Plot title
    save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    counts = target.value_counts()
    labels = ['Normal' if x == -1 else 'Failure' for x in counts.index]
    colors = ['lightblue', 'lightcoral']
    
    plt.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(title)
    
    # Add count information
    total = len(target)
    plt.figtext(0.02, 0.02, f'Total samples: {total}\nNormal: {counts.get(-1, 0)}\nFailure: {counts.get(1, 0)}',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], 
                   format=VISUALIZATION_CONFIG['plot_format'])
    
    plt.show()

def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """
    Plot training history (for models that support it)
    
    Parameters:
    history (Dict[str, List[float]]): Training history metrics
    save_path (str): Path to save the plot
    """
    if not history:
        print("No training history available to plot")
        return
    
    fig, axes = plt.subplots(1, len(history), figsize=(15, 5))
    if len(history) == 1:
        axes = [axes]
    
    for i, (metric, values) in enumerate(history.items()):
        axes[i].plot(values)
        axes[i].set_title(f'Training {metric.capitalize()}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], 
                   format=VISUALIZATION_CONFIG['plot_format'])
    
    plt.show()

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def save_model_with_metadata(model: Any, model_name: str, metadata: Dict[str, Any],
                            directory: str = None) -> str:
    """
    Save model with metadata
    
    Parameters:
    model: Trained model object
    model_name (str): Name of the model
    metadata (Dict[str, Any]): Model metadata
    directory (str): Directory to save the model
    
    Returns:
    str: Path to saved model
    """
    if directory is None:
        directory = STORAGE_CONFIG['models_directory']
    
    os.makedirs(directory, exist_ok=True)
    
    # Save model
    model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    model_path = os.path.join(directory, model_filename)
    
    if STORAGE_CONFIG['model_format'] == 'joblib':
        joblib.dump(model, model_path)
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save metadata
    metadata_filename = f"{model_name.replace(' ', '_').lower()}_metadata.json"
    metadata_path = os.path.join(directory, metadata_filename)
    
    # Add timestamp and version info
    metadata.update({
        'saved_at': datetime.now().isoformat(),
        'model_file': model_filename,
        'model_format': STORAGE_CONFIG['model_format']
    })
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path

def load_model_with_metadata(model_name: str, directory: str = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model with metadata
    
    Parameters:
    model_name (str): Name of the model
    directory (str): Directory containing the model
    
    Returns:
    Tuple[Any, Dict[str, Any]]: Loaded model and metadata
    """
    if directory is None:
        directory = STORAGE_CONFIG['models_directory']
    
    # Load model
    model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
    model_path = os.path.join(directory, model_filename)
    
    if STORAGE_CONFIG['model_format'] == 'joblib':
        model = joblib.load(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    # Load metadata
    metadata_filename = f"{model_name.replace(' ', '_').lower()}_metadata.json"
    metadata_path = os.path.join(directory, metadata_filename)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of model results
    
    Parameters:
    results (Dict[str, Dict[str, float]]): Model results dictionary
    
    Returns:
    pd.DataFrame: Comparison table
    """
    df = pd.DataFrame(results).T
    df = df.round(4)
    df['accuracy_pct'] = df['accuracy'] * 100
    df = df.sort_values('f1_score', ascending=False)
    
    return df

def get_model_complexity_score(model: Any) -> float:
    """
    Calculate a complexity score for the model
    
    Parameters:
    model: Trained model object
    
    Returns:
    float: Complexity score (0-1, where 1 is most complex)
    """
    model_type = type(model).__name__
    
    complexity_scores = {
        'RandomForestClassifier': 0.7,
        'SVC': 0.8,
        'LogisticRegression': 0.3,
        'KNeighborsClassifier': 0.4,
        'MLPClassifier': 0.9,
        'GradientBoostingClassifier': 0.8,
        'XGBClassifier': 0.8,
    }
    
    base_score = complexity_scores.get(model_type, 0.5)
    
    # Adjust based on model parameters
    if hasattr(model, 'n_estimators'):
        base_score += min(model.n_estimators / 1000, 0.2)
    
    if hasattr(model, 'max_depth') and model.max_depth:
        base_score += min(model.max_depth / 50, 0.1)
    
    return min(base_score, 1.0)

# =============================================================================
# PERFORMANCE METRICS UTILITIES
# =============================================================================

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics
    
    Parameters:
    y_true (np.ndarray): True labels
    y_pred (np.ndarray): Predicted labels
    y_pred_proba (np.ndarray): Prediction probabilities (optional)
    
    Returns:
    Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, matthews_corrcoef)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'f_measure': f1_score(y_true, y_pred, average='weighted'),  # Same as F1
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
    }
    
    # Add AUC if probabilities are available
    if y_pred_proba is not None:
        try:
            # For binary classification with labels -1, 1
            y_true_binary = (y_true == 1).astype(int)
            if y_pred_proba.ndim == 2:
                auc_score = roc_auc_score(y_true_binary, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_true_binary, y_pred_proba)
            metrics['auc_roc'] = auc_score
        except:
            pass
    
    return metrics

def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calculate confusion matrix components
    
    Parameters:
    y_true (np.ndarray): True labels
    y_pred (np.ndarray): Predicted labels
    
    Returns:
    Dict[str, int]: Confusion matrix components
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # For cases where one class is missing
        tn = fp = fn = tp = 0
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, true_label in enumerate(unique_labels):
            for j, pred_label in enumerate(unique_labels):
                count = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
                if true_label == -1 and pred_label == -1:
                    tn = count
                elif true_label == -1 and pred_label == 1:
                    fp = count
                elif true_label == 1 and pred_label == -1:
                    fn = count
                elif true_label == 1 and pred_label == 1:
                    tp = count
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

# =============================================================================
# COST-BENEFIT ANALYSIS UTILITIES
# =============================================================================

def calculate_maintenance_costs(cm_metrics: Dict[str, int], cost_config: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate maintenance costs based on confusion matrix
    
    Parameters:
    cm_metrics (Dict[str, int]): Confusion matrix metrics
    cost_config (Dict[str, float]): Cost configuration
    
    Returns:
    Dict[str, float]: Cost breakdown
    """
    tp = cm_metrics['true_positives']
    fp = cm_metrics['false_positives']
    fn = cm_metrics['false_negatives']
    tn = cm_metrics['true_negatives']
    
    # Calculate costs
    cost_prevented_failures = tp * cost_config['planned_maintenance_cost']
    cost_missed_failures = fn * cost_config['unplanned_downtime_cost']
    cost_false_alarms = fp * cost_config['false_positive_cost']
    cost_normal_operation = 0  # No cost for correct normal predictions
    
    total_cost_with_pm = cost_prevented_failures + cost_missed_failures + cost_false_alarms
    total_failures = tp + fn
    cost_without_pm = total_failures * cost_config['unplanned_downtime_cost']
    
    savings = cost_without_pm - total_cost_with_pm
    savings_percentage = (savings / cost_without_pm * 100) if cost_without_pm > 0 else 0
    
    return {
        'cost_prevented_failures': cost_prevented_failures,
        'cost_missed_failures': cost_missed_failures,
        'cost_false_alarms': cost_false_alarms,
        'total_cost_with_pm': total_cost_with_pm,
        'cost_without_pm': cost_without_pm,
        'total_savings': savings,
        'savings_percentage': savings_percentage
    }

def generate_cost_benefit_report(results: Dict[str, Dict[str, float]], 
                               cost_config: Dict[str, float]) -> str:
    """
    Generate a comprehensive cost-benefit analysis report
    
    Parameters:
    results (Dict[str, Dict[str, float]]): Model performance results
    cost_config (Dict[str, float]): Cost configuration
    
    Returns:
    str: Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("COST-BENEFIT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("Cost Assumptions:")
    report.append(f"- Unplanned downtime cost: ${cost_config['unplanned_downtime_cost']:,}")
    report.append(f"- Planned maintenance cost: ${cost_config['planned_maintenance_cost']:,}")
    report.append(f"- False positive cost: ${cost_config['false_positive_cost']:,}")
    report.append("")
    
    for model_name, metrics in results.items():
        report.append(f"Model: {model_name}")
        report.append("-" * 40)
        
        # This would need actual y_true and y_pred to calculate costs
        # For now, we'll provide a template
        report.append(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        report.append(f"F1 Score: {metrics['f1_score']:.3f}")
        report.append("")
    
    report.append("Recommendations:")
    report.append("- Deploy the highest performing model for production use")
    report.append("- Implement real-time monitoring with automated alerts")
    report.append("- Regular model retraining to maintain performance")
    report.append("- Integration with maintenance management systems")
    
    return "\n".join(report)

# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def generate_model_summary_report(model_name: str, metrics: Dict[str, float],
                                metadata: Dict[str, Any] = None) -> str:
    """
    Generate a summary report for a single model
    
    Parameters:
    model_name (str): Name of the model
    metrics (Dict[str, float]): Performance metrics
    metadata (Dict[str, Any]): Model metadata
    
    Returns:
    str: Formatted report
    """
    report = []
    report.append(f"MODEL SUMMARY REPORT: {model_name}")
    report.append("=" * 60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("Performance Metrics:")
    report.append("-" * 20)
    for metric, value in metrics.items():
        if metric == 'accuracy':
            report.append(f"- {metric.capitalize()}: {value*100:.2f}%")
        else:
            report.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
    report.append("")
    
    if metadata:
        report.append("Model Metadata:")
        report.append("-" * 15)
        for key, value in metadata.items():
            if key not in ['saved_at', 'model_file']:
                report.append(f"- {key.replace('_', ' ').title()}: {value}")
        report.append("")
    
    # Performance interpretation
    report.append("Performance Interpretation:")
    report.append("-" * 25)
    accuracy = metrics.get('accuracy', 0)
    f1_score = metrics.get('f1_score', 0)
    
    if accuracy >= 0.95 and f1_score >= 0.95:
        report.append("🟢 Excellent performance - Ready for production deployment")
    elif accuracy >= 0.90 and f1_score >= 0.90:
        report.append("🟡 Good performance - Consider further optimization")
    elif accuracy >= 0.80 and f1_score >= 0.80:
        report.append("🟠 Moderate performance - Requires improvement")
    else:
        report.append("🔴 Poor performance - Not recommended for deployment")
    
    return "\n".join(report)

def export_results_to_csv(results: Dict[str, Dict[str, float]], 
                         filename: str = None) -> str:
    """
    Export model results to CSV file
    
    Parameters:
    results (Dict[str, Dict[str, float]]): Model results
    filename (str): Output filename
    
    Returns:
    str: Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_results_{timestamp}.csv"
    
    # Create results directory if it doesn't exist
    results_dir = STORAGE_CONFIG['results_directory']
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results).T
    df.index.name = 'model'
    df.to_csv(filepath)
    
    return filepath

# =============================================================================
# SYSTEM HEALTH UTILITIES
# =============================================================================

def check_system_health() -> Dict[str, Any]:
    """
    Check system health and dependencies
    
    Returns:
    Dict[str, Any]: System health status
    """
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'directories': {},
        'dependencies': {},
        'memory_usage': {},
        'disk_space': {}
    }
    
    # Check directories
    required_dirs = [
        STORAGE_CONFIG['models_directory'],
        STORAGE_CONFIG['logs_directory'],
        STORAGE_CONFIG['results_directory'],
        STORAGE_CONFIG['data_directory']
    ]
    
    for directory in required_dirs:
        health_status['directories'][directory] = {
            'exists': os.path.exists(directory),
            'writable': os.access(directory, os.W_OK) if os.path.exists(directory) else False
        }
    
    # Check dependencies
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'imbalanced-learn', 'joblib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            health_status['dependencies'][package] = 'OK'
        except ImportError:
            health_status['dependencies'][package] = 'MISSING'
    
    # Check memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        health_status['memory_usage'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_percentage': memory.percent
        }
    except ImportError:
        health_status['memory_usage'] = 'psutil not available'
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        health_status['disk_space'] = {
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'free_gb': round(disk_usage.free / (1024**3), 2),
            'used_percentage': round((disk_usage.used / disk_usage.total) * 100, 2)
        }
    except:
        health_status['disk_space'] = 'Unable to check'
    
    return health_status

def print_system_health():
    """Print system health status in a readable format"""
    health = check_system_health()
    
    print("🔍 SYSTEM HEALTH CHECK")
    print("=" * 40)
    print(f"Timestamp: {health['timestamp']}")
    print()
    
    print("📁 Directories:")
    for directory, status in health['directories'].items():
        exists_icon = "✅" if status['exists'] else "❌"
        writable_icon = "✅" if status['writable'] else "❌"
        print(f"  {directory}: {exists_icon} Exists {writable_icon} Writable")
    print()
    
    print("📦 Dependencies:")
    for package, status in health['dependencies'].items():
        icon = "✅" if status == 'OK' else "❌"
        print(f"  {package}: {icon} {status}")
    print()
    
    if isinstance(health['memory_usage'], dict):
        print("💾 Memory Usage:")
        mem = health['memory_usage']
        print(f"  Total: {mem['total_gb']} GB")
        print(f"  Available: {mem['available_gb']} GB")
        print(f"  Used: {mem['used_percentage']}%")
        print()
    
    if isinstance(health['disk_space'], dict):
        print("💿 Disk Space:")
        disk = health['disk_space']
        print(f"  Total: {disk['total_gb']} GB")
        print(f"  Free: {disk['free_gb']} GB")
        print(f"  Used: {disk['used_percentage']}%")

# =============================================================================
# MAIN UTILITY FUNCTIONS
# =============================================================================

def initialize_system() -> bool:
    """
    Initialize the predictive maintenance system
    
    Returns:
    bool: True if initialization successful
    """
    try:
        # Setup directories
        from config import create_directories
        create_directories()
        
        # Setup logging
        setup_logging()
        
        # Setup plotting
        setup_plotting_style()
        
        print("✅ System initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Predictive Maintenance Utilities")
    print("=" * 40)
    
    # Initialize system
    if initialize_system():
        # Run system health check
        print_system_health()
        
        print("\n🚀 Utilities module ready for use!")
    else:
        print("\n❌ Utilities module initialization failed!")