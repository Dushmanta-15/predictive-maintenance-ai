#!/usr/bin/env python3
"""
Basic Usage Example for AI-Driven Predictive Maintenance System

This script demonstrates the basic functionality of the predictive maintenance system
with minimal code for quick testing and evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictive_maintenance import PredictiveMaintenanceSystem
import numpy as np

def basic_example():
    """Run a basic example of the predictive maintenance system"""
    
    print("🔧 AI-Driven Predictive Maintenance - Basic Example")
    print("=" * 55)
    
    # Step 1: Initialize the system
    print("\n1. Initializing system...")
    pm_system = PredictiveMaintenanceSystem()
    
    # Step 2: Load synthetic data
    print("2. Loading synthetic SECOM-like dataset...")
    data, target = pm_system.load_data(synthetic=True)
    print(f"   Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Step 3: Preprocess data
    print("3. Preprocessing data...")
    X_train, X_test, y_train, y_test = pm_system.preprocess_data()
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Step 4: Train models
    print("4. Training machine learning models...")
    pm_system.initialize_models()
    pm_system.train_models(X_train, y_train)
    print("   ✅ All models trained successfully!")
    
    # Step 5: Evaluate models
    print("5. Evaluating model performance...")
    results = pm_system.evaluate_models(X_test, y_test)
    
    # Step 6: Display results
    print("\n6. Performance Results:")
    pm_system.display_results()
    
    # Step 7: Run real-time simulation
    print("\n7. Running real-time monitoring simulation...")
    accuracy, alerts = pm_system.real_time_monitoring_simulation(
        X_test, y_test, 'Random Forest', n_iterations=20
    )
    
    print(f"\n📊 Simulation Results:")
    print(f"   Real-time accuracy: {accuracy*100:.1f}%")
    print(f"   Alerts triggered: {alerts}")
    
    # Step 8: Save models
    print("\n8. Saving trained models...")
    pm_system.save_models('./models')
    print("   ✅ Models saved successfully!")
    
    # Step 9: Test single prediction
    print("\n9. Testing single sample prediction...")
    sample_idx = np.random.choice(len(X_test))
    sample = X_test[sample_idx]
    true_label = y_test.iloc[sample_idx]
    
    prediction, probability = pm_system.predict_single_sample(sample, 'Random Forest')
    
    print(f"   Sample prediction:")
    print(f"   True label: {'Failure' if true_label == 1 else 'Normal'}")
    print(f"   Predicted: {'Failure' if prediction == 1 else 'Normal'}")
    if probability is not None:
        print(f"   Confidence: {max(probability)*100:.1f}%")
    
    print("\n🎉 Basic example completed successfully!")
    print("\nNext steps:")
    print("- Check the saved models in ./models directory")
    print("- Run the full demo notebook for detailed analysis")
    print("- Integrate with your own sensor data")

def quick_test():
    """Quick test to verify system functionality"""
    
    print("🚀 Quick System Test")
    print("=" * 20)
    
    try:
        # Initialize and run minimal test
        pm_system = PredictiveMaintenanceSystem()
        data, target = pm_system.load_data(synthetic=True)
        X_train, X_test, y_train, y_test = pm_system.preprocess_data()
        
        # Train only Random Forest for quick test
        pm_system.models['Random Forest'] = pm_system.initialize_models()['Random Forest']
        pm_system.models['Random Forest'].fit(X_train, y_train)
        pm_system.is_trained = True
        
        # Quick evaluation
        from sklearn.metrics import accuracy_score
        predictions = pm_system.models['Random Forest'].predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"✅ System test passed!")
        print(f"   Random Forest accuracy: {accuracy*100:.1f}%")
        print(f"   System is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Driven Predictive Maintenance - Basic Usage')
    parser.add_argument('--mode', choices=['basic', 'test'], default='basic',
                       help='Run mode: basic example or quick test')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        basic_example()
    elif args.mode == 'test':
        success = quick_test()
        sys.exit(0 if success else 1)
