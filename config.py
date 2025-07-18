"""
Configuration file for AI-Driven Predictive Maintenance System

This file contains all the configurable parameters for the predictive maintenance system.
Modify these values to customize the system behavior for your specific use case.
"""

import os

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Dataset parameters
DATA_CONFIG = {
    'synthetic_data': {
        'n_samples': 1567,           # Number of samples (similar to SECOM)
        'n_features': 590,           # Number of features (similar to SECOM)
        'missing_rate': 0.05,        # Percentage of missing values
        'failure_rate': 0.07,        # Percentage of failure samples
        'random_seed': 42,           # Random seed for reproducibility
    },
    'real_data': {
        'path': './data/secom_dataset.csv',
        'target_column': -1,         # Index of target column (-1 for last column)
        'separator': ',',            # CSV separator
        'encoding': 'utf-8',         # File encoding
    }
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Machine Learning model parameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'random_state': 42,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'n_jobs': -1,               # Use all available cores
    },
    'svm': {
        'kernel': 'linear',
        'class_weight': 'balanced',
        'random_state': 42,
        'C': 1.0,
        'gamma': 'scale',
        'probability': True,        # Enable probability estimates
    },
    'logistic_regression': {
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'liblinear',
        'penalty': 'l2',
        'C': 1.0,
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'uniform',
        'algorithm': 'auto',
        'metric': 'minkowski',
        'p': 2,                     # For minkowski metric
    }
}

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    'missing_values': {
        'strategy': 'mean',         # 'mean', 'median', 'most_frequent', 'constant'
        'fill_value': None,         # Used when strategy='constant'
    },
    'scaling': {
        'method': 'standard',       # 'standard', 'minmax', 'robust', 'none'
        'with_mean': True,
        'with_std': True,
    },
    'smote': {
        'random_state': 42,
        'k_neighbors': 5,
        'sampling_strategy': 'auto', # 'auto', 'minority', 'not majority', 'not minority', 'all'
    },
    'train_test_split': {
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True,           # Maintain class distribution
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Model evaluation parameters
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'f_measure'],
    'cross_validation': {
        'enabled': False,           # Enable cross-validation
        'cv_folds': 5,
        'scoring': 'f1_weighted',
    },
    'confusion_matrix': {
        'normalize': None,          # None, 'true', 'pred', 'all'
        'labels': [-1, 1],          # Class labels
    }
}

# =============================================================================
# REAL-TIME MONITORING CONFIGURATION
# =============================================================================

# Real-time monitoring parameters
MONITORING_CONFIG = {
    'simulation': {
        'default_iterations': 50,
        'default_model': 'Random Forest',
        'alert_threshold': 0.5,     # Probability threshold for alerts
        'sampling_frequency': 1.0,  # Seconds between samples
    },
    'alerts': {
        'failure_message': "⚠️  ALERT - Equipment failure predicted!",
        'normal_message': "✅ Equipment operating normally",
        'log_predictions': True,
        'save_alerts': True,
    }
}

# =============================================================================
# COST-BENEFIT ANALYSIS CONFIGURATION
# =============================================================================

# Cost parameters for economic analysis
COST_CONFIG = {
    'unplanned_downtime_cost': 10000,    # Cost per unplanned failure ($)
    'planned_maintenance_cost': 2000,     # Cost per planned maintenance ($)
    'false_positive_cost': 500,           # Cost of unnecessary maintenance ($)
    'sensor_cost': 100,                   # Cost per sensor ($)
    'system_deployment_cost': 50000,      # One-time deployment cost ($)
    'annual_maintenance_cost': 5000,      # Annual system maintenance cost ($)
}

# =============================================================================
# STORAGE AND PERSISTENCE CONFIGURATION
# =============================================================================

# File paths and storage configuration
STORAGE_CONFIG = {
    'models_directory': './models',
    'logs_directory': './logs',
    'results_directory': './results',
    'data_directory': './data',
    'model_format': 'joblib',       # 'joblib', 'pickle'
    'compression': True,            # Compress saved models
    'backup_models': True,          # Keep backup of previous models
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging parameters
LOGGING_CONFIG = {
    'level': 'INFO',                # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': './logs/predictive_maintenance.log',
    'max_log_size': 10485760,       # 10MB
    'backup_count': 5,
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Plotting and visualization parameters
VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8',        # Matplotlib style
    'color_palette': 'husl',        # Seaborn color palette
    'figure_size': (12, 8),         # Default figure size
    'dpi': 100,                     # Figure resolution
    'save_plots': True,             # Save plots to file
    'plot_format': 'png',           # png, jpg, pdf, svg
    'plot_directory': './plots',
}

# =============================================================================
# ADVANCED FEATURES CONFIGURATION
# =============================================================================

# Advanced features (experimental)
ADVANCED_CONFIG = {
    'feature_selection': {
        'enabled': False,
        'method': 'variance_threshold',  # 'variance_threshold', 'univariate', 'rfe'
        'threshold': 0.01,
        'n_features': 100,
    },
    'ensemble_methods': {
        'enabled': False,
        'voting_classifier': True,
        'stacking_classifier': False,
        'weights': None,            # Weights for voting classifier
    },
    'hyperparameter_tuning': {
        'enabled': False,
        'method': 'grid_search',    # 'grid_search', 'random_search', 'bayesian'
        'cv_folds': 3,
        'n_iter': 50,               # For random search
    }
}

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

# Deployment and production parameters
DEPLOYMENT_CONFIG = {
    'api': {
        'host': '0.0.0.0',
        'port': 8000,
        'debug': False,
        'workers': 4,
    },
    'database': {
        'enabled': False,
        'type': 'sqlite',           # sqlite, postgresql, mysql
        'path': './database.db',
        'host': 'localhost',
        'port': 5432,
        'username': '',
        'password': '',
        'database_name': 'predictive_maintenance',
    },
    'cache': {
        'enabled': False,
        'type': 'redis',            # redis, memcached
        'host': 'localhost',
        'port': 6379,
        'ttl': 3600,                # Time to live in seconds
    }
}

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Security parameters
SECURITY_CONFIG = {
    'api_key_required': False,
    'rate_limiting': {
        'enabled': False,
        'requests_per_minute': 60,
    },
    'data_encryption': {
        'enabled': False,
        'algorithm': 'AES-256',
    },
    'audit_logging': True,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(section=None):
    """
    Get configuration for a specific section or all configurations
    
    Parameters:
    section (str): Configuration section name
    
    Returns:
    dict: Configuration dictionary
    """
    configs = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'cost': COST_CONFIG,
        'storage': STORAGE_CONFIG,
        'logging': LOGGING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'deployment': DEPLOYMENT_CONFIG,
        'security': SECURITY_CONFIG,
    }
    
    if section:
        return configs.get(section, {})
    return configs

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        STORAGE_CONFIG['models_directory'],
        STORAGE_CONFIG['logs_directory'],
        STORAGE_CONFIG['results_directory'],
        STORAGE_CONFIG['data_directory'],
        VISUALIZATION_CONFIG['plot_directory'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Validate data configuration
    if DATA_CONFIG['synthetic_data']['failure_rate'] >= 1.0:
        errors.append("Failure rate must be less than 1.0")
    
    # Validate model configuration
    if MODEL_CONFIG['random_forest']['n_estimators'] <= 0:
        errors.append("Random Forest n_estimators must be positive")
    
    # Validate preprocessing configuration
    if PREPROCESSING_CONFIG['train_test_split']['test_size'] >= 1.0:
        errors.append("Test size must be less than 1.0")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    return True

# Initialize directories when module is imported
if __name__ != "__main__":
    try:
        create_directories()
        validate_config()
    except Exception as e:
        print(f"Warning: Configuration initialization failed: {e}")

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("🔧 Predictive Maintenance System Configuration")
    print("=" * 50)
    
    # Create directories
    create_directories()
    print("✅ Directories created")
    
    # Validate configuration
    try:
        validate_config()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
    
    # Display key configuration
    print(f"\nKey Configuration Settings:")
    print(f"- Dataset: {DATA_CONFIG['synthetic_data']['n_samples']} samples, "
          f"{DATA_CONFIG['synthetic_data']['n_features']} features")
    print(f"- Models: {', '.join(MODEL_CONFIG.keys())}")
    print(f"- Test size: {PREPROCESSING_CONFIG['train_test_split']['test_size']*100:.0f}%")
    print(f"- Models directory: {STORAGE_CONFIG['models_directory']}")
    print(f"- Logging level: {LOGGING_CONFIG['level']}")
    
    print("\n🚀 Configuration ready for use!")
