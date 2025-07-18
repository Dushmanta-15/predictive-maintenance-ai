import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceSystem:
    """
    AI-Driven Predictive Maintenance System for Industrial Equipment
    
    This system implements machine learning models to predict equipment failures
    using sensor data, as described in the research paper by Kalita & Das.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.smote = SMOTE(random_state=42)
        self.results = {}
        self.is_trained = False
        
    def load_data(self, data_path=None, synthetic=False):
        """
        Load the SECOM dataset or generate synthetic data for demonstration
        
        Parameters:
        data_path (str): Path to the SECOM dataset
        synthetic (bool): If True, generates synthetic data similar to SECOM
        """
        if synthetic or data_path is None:
            print("Generating synthetic SECOM-like dataset...")
            # Generate synthetic data similar to SECOM dataset characteristics
            np.random.seed(42)
            n_samples = 1567
            n_features = 590
            
            # Create synthetic sensor data with some correlation structure
            data = np.random.normal(0, 1, (n_samples, n_features))
            
            # Add some missing values (similar to SECOM)
            missing_mask = np.random.random((n_samples, n_features)) < 0.05
            data[missing_mask] = np.nan
            
            # Create imbalanced target (similar to SECOM)
            # Most samples are normal (-1), few are failures (1)
            target = np.random.choice([-1, 1], n_samples, p=[0.93, 0.07])
            
            # Create feature names
            feature_names = [f'sensor_{i}' for i in range(n_features)]
            
            self.data = pd.DataFrame(data, columns=feature_names)
            self.target = pd.Series(target, name='target')
            
            print(f"Generated synthetic dataset with {n_samples} samples and {n_features} features")
            print(f"Target distribution: {pd.Series(target).value_counts()}")
            
        else:
            # Load actual SECOM dataset
            print(f"Loading SECOM dataset from {data_path}")
            self.data = pd.read_csv(data_path)
            # Assuming the last column is the target
            self.target = self.data.iloc[:, -1]
            self.data = self.data.iloc[:, :-1]
            
        return self.data, self.target
    
    def preprocess_data(self):
        """
        Preprocess the data following the methodology in the paper:
        1. Handle missing values using mean imputation
        2. Normalize features using StandardScaler
        3. Address class imbalance using SMOTE
        """
        print("Preprocessing data...")
        
        # Handle missing values
        print("Handling missing values...")
        self.data_processed = pd.DataFrame(
            self.imputer.fit_transform(self.data),
            columns=self.data.columns
        )
        
        # Split data before SMOTE (to avoid data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_processed, self.target, 
            test_size=0.2, random_state=42, stratify=self.target
        )
        
        # Normalize features
        print("Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE to training data only
        print("Applying SMOTE to address class imbalance...")
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_scaled, y_train)
        
        print(f"Original training set distribution: {pd.Series(y_train).value_counts()}")
        print(f"Balanced training set distribution: {pd.Series(y_train_balanced).value_counts()}")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def initialize_models(self):
        """
        Initialize the machine learning models as described in the paper
        """
        print("Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ),
            'SVM': SVC(
                kernel='linear',
                class_weight='balanced',
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            )
        }
        
        return self.models
    
    def train_models(self, X_train, y_train):
        """
        Train all models and store them
        """
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
        self.is_trained = True
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models and return performance metrics
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        print("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # F-measure (same as F1 for binary classification)
            f_measure = f1
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'f_measure': f_measure
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        self.results = results
        return results
    
    def display_results(self):
        """
        Display results in a formatted table similar to the paper
        """
        if not self.results:
            print("No results to display. Please run evaluation first.")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON OF MACHINE LEARNING MODELS")
        print("="*80)
        print(f"{'Model':<20} {'Accuracy (%)':<12} {'Precision':<10} {'Recall':<8} {'F1 Score':<10} {'F-Measure':<10}")
        print("-"*80)
        
        for name, metrics in self.results.items():
            print(f"{name:<20} {metrics['accuracy']*100:<12.2f} {metrics['precision']:<10.2f} "
                  f"{metrics['recall']:<8.2f} {metrics['f1_score']:<10.2f} {metrics['f_measure']:<10.2f}")
        
        print("-"*80)
    
    def plot_results(self):
        """
        Create visualizations of model performance
        """
        if not self.results:
            print("No results to plot. Please run evaluation first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            values = [self.results[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def real_time_monitoring_simulation(self, X_test, y_test, model_name='Random Forest', n_iterations=50):
        """
        Simulate real-time monitoring as described in the paper
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before simulation")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        print(f"\nStarting real-time monitoring simulation with {model_name}...")
        print(f"Monitoring {n_iterations} samples...")
        
        # Randomly sample from test set
        indices = np.random.choice(len(X_test), n_iterations, replace=False)
        
        correct_predictions = 0
        alerts_triggered = 0
        
        for i, idx in enumerate(indices):
            sample = X_test[idx].reshape(1, -1)
            true_label = y_test.iloc[idx]
            
            # Make prediction
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0] if hasattr(model, 'predict_proba') else None
            
            # Check if prediction is correct
            if prediction == true_label:
                correct_predictions += 1
            
            # Trigger alert if failure is predicted
            if prediction == 1:
                alerts_triggered += 1
                prob_text = f" (Probability: {probability[1]:.3f})" if probability is not None else ""
                print(f"⚠️  ALERT - Iteration {i+1}: Equipment failure predicted!{prob_text}")
            
            # Show progress every 10 iterations
            if (i + 1) % 10 == 0:
                accuracy = correct_predictions / (i + 1)
                print(f"Progress: {i+1}/{n_iterations} - Current accuracy: {accuracy:.3f}")
        
        final_accuracy = correct_predictions / n_iterations
        print(f"\nReal-time monitoring simulation completed!")
        print(f"Final accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
        print(f"Alerts triggered: {alerts_triggered}")
        
        return final_accuracy, alerts_triggered
    
    def save_models(self, directory='./models'):
        """
        Save trained models and preprocessing components
        """
        if not self.is_trained:
            print("No trained models to save")
            return
        
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filename = f"{directory}/{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} model to {filename}")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")
        joblib.dump(self.imputer, f"{directory}/imputer.pkl")
        
        print(f"All models and preprocessing components saved to {directory}")
    
    def load_models(self, directory='./models'):
        """
        Load trained models and preprocessing components
        """
        import os
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
        
        # Load models
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'SVM': 'svm_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'KNN': 'knn_model.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = f"{directory}/{filename}"
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"Loaded {name} model from {filepath}")
        
        # Load preprocessing components
        scaler_path = f"{directory}/scaler.pkl"
        imputer_path = f"{directory}/imputer.pkl"
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded scaler")
        
        if os.path.exists(imputer_path):
            self.imputer = joblib.load(imputer_path)
            print("Loaded imputer")
        
        self.is_trained = True
        print("Models loaded successfully!")
    
    def predict_single_sample(self, sample_data, model_name='Random Forest'):
        """
        Predict failure for a single sample
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Preprocess the sample
        sample_processed = self.imputer.transform([sample_data])
        sample_scaled = self.scaler.transform(sample_processed)
        
        # Make prediction
        prediction = model.predict(sample_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(sample_scaled)[0]
            return prediction, probability
        else:
            return prediction, None

# Example usage and demonstration
def main():
    """
    Main function to demonstrate the predictive maintenance system
    """
    print("AI-Driven Predictive Maintenance System")
    print("Based on the research by Kalita & Das")
    print("="*50)
    
    # Initialize the system
    pm_system = PredictiveMaintenanceSystem()
    
    # Load data (using synthetic data for demonstration)
    data, target = pm_system.load_data(synthetic=True)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = pm_system.preprocess_data()
    
    # Initialize and train models
    pm_system.initialize_models()
    pm_system.train_models(X_train, y_train)
    
    # Evaluate models
    results = pm_system.evaluate_models(X_test, y_test)
    
    # Display results
    pm_system.display_results()
    
    # Plot results
    pm_system.plot_results()
    
    # Real-time monitoring simulation
    pm_system.real_time_monitoring_simulation(X_test, y_test, 'Random Forest', 50)
    
    # Save models
    pm_system.save_models()
    
    # Example of single prediction
    print("\n" + "="*50)
    print("SINGLE SAMPLE PREDICTION EXAMPLE")
    print("="*50)
    
    # Get a random sample from test set
    sample_idx = np.random.choice(len(X_test))
    sample = X_test[sample_idx]
    true_label = y_test.iloc[sample_idx]
    
    # Make prediction
    prediction, probability = pm_system.predict_single_sample(sample, 'Random Forest')
    
    print(f"Sample prediction:")
    print(f"True label: {true_label} ({'Failure' if true_label == 1 else 'Normal'})")
    print(f"Predicted: {prediction} ({'Failure' if prediction == 1 else 'Normal'})")
    if probability is not None:
        print(f"Probability: Normal={probability[0]:.3f}, Failure={probability[1]:.3f}")
    
    print("\nSystem demonstration completed!")

if __name__ == "__main__":
    main()
