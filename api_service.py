#!/usr/bin/env python3
"""
REST API Service for AI-Driven Predictive Maintenance System

This module provides a REST API interface for the predictive maintenance system,
allowing external systems to interact with the trained models for real-time predictions.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


# Flask imports
try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

# Import our predictive maintenance system
from predictive_maintenance import PredictiveMaintenanceSystem
from config import get_config
from utils import setup_logging, calculate_comprehensive_metrics

# =============================================================================
# API SERVICE CLASS
# =============================================================================

class PredictiveMaintenanceAPI:
    """
    REST API service for predictive maintenance system
    """
    
    def __init__(self, model_directory: str = './models'):
        """
        Initialize the API service
        
        Parameters:
        model_directory (str): Directory containing trained models
        """
        self.model_directory = model_directory
        self.pm_system = PredictiveMaintenanceSystem()
        self.logger = setup_logging()
        self.is_loaded = False
        
        # Load models if available
        self.load_models()
    
    def load_models(self) -> bool:
        """
        Load trained models from disk
        
        Returns:
        bool: True if models loaded successfully
        """
        try:
            if os.path.exists(self.model_directory):
                self.pm_system.load_models(self.model_directory)
                self.is_loaded = True
                self.logger.info("Models loaded successfully")
                return True
            else:
                self.logger.warning(f"Model directory {self.model_directory} not found")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def predict_failure(self, sensor_data: List[float], model_name: str = 'Random Forest') -> Dict[str, Any]:
        """
        Predict equipment failure for given sensor data
        
        Parameters:
        sensor_data (List[float]): Sensor readings
        model_name (str): Name of model to use for prediction
        
        Returns:
        Dict[str, Any]: Prediction result
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Please ensure models are trained and saved.")
        
        try:
            # Make prediction
            prediction, probability = self.pm_system.predict_single_sample(
                sensor_data, model_name
            )
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Equipment Failure' if prediction == 1 else 'Normal Operation',
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'confidence': None,
                'alert_level': self._get_alert_level(prediction, probability)
            }
            
            if probability is not None:
                result['confidence'] = float(max(probability))
                result['probability_normal'] = float(probability[0])
                result['probability_failure'] = float(probability[1])
            
            # Log prediction
            self.logger.info(f"Prediction made: {result['prediction_label']} with {model_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def _get_alert_level(self, prediction: int, probability: Optional[np.ndarray]) -> str:
        """
        Determine alert level based on prediction and probability
        
        Parameters:
        prediction (int): Predicted class
        probability (Optional[np.ndarray]): Prediction probabilities
        
        Returns:
        str: Alert level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if prediction == -1:
            return "LOW"
        
        if probability is not None:
            failure_prob = probability[1]
            if failure_prob >= 0.9:
                return "CRITICAL"
            elif failure_prob >= 0.7:
                return "HIGH"
            else:
                return "MEDIUM"
        
        return "HIGH"  # Default for failure prediction without probability
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
        Dict[str, Any]: Model information
        """
        if not self.is_loaded:
            return {'models_loaded': False, 'available_models': []}
        
        return {
            'models_loaded': True,
            'available_models': list(self.pm_system.models.keys()),
            'model_directory': self.model_directory,
            'system_status': 'Ready'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform system health check
        
        Returns:
        Dict[str, Any]: Health status
        """
        from utils import check_system_health
        
        health = check_system_health()
        health.update({
            'api_status': 'Running',
            'models_loaded': self.is_loaded,
            'available_models': len(self.pm_system.models) if self.is_loaded else 0
        })
        
        return health

# =============================================================================
# FLASK APPLICATION
# =============================================================================

if FLASK_AVAILABLE:
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Initialize API service
    api_service = PredictiveMaintenanceAPI()
    
    # Simple HTML template for the web interface
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictive Maintenance API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #fff; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 AI-Driven Predictive Maintenance API</h1>
            <p>RESTful API for equipment failure prediction using machine learning models.</p>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/</code>
                <p>This documentation page</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/health</code>
                <p>System health check and status information</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/models</code>
                <p>Get information about loaded models</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/predict</code>
                <p>Predict equipment failure based on sensor data</p>
                <p><strong>Request Body:</strong></p>
                <pre>{
      "sensor_data": [0.1, 0.2, ..., 0.9],
      "model_name": "Random Forest"  // optional
    }</pre>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/batch_predict</code>
                <p>Predict multiple samples at once</p>
                <p><strong>Request Body:</strong></p>
                <pre>{
      "samples": [
          [0.1, 0.2, ..., 0.9],
          [0.2, 0.3, ..., 0.8]
      ],
      "model_name": "Random Forest"  // optional
    }</pre>
            </div>
            
            <h2>Status</h2>
            <p id="status">Loading...</p>
            
            <script>
                fetch('/models')
                    .then(response => response.json())
                    .then(data => {
                        const status = data.models_loaded ? 
                            `✅ ${data.available_models.length} models loaded: ${data.available_models.join(', ')}` :
                            '❌ No models loaded';
                        document.getElementById('status').textContent = status;
                    })
                    .catch(error => {
                        document.getElementById('status').textContent = '❌ Error loading status';
                    });
            </script>
        </div>
    </body>
    </html>
    """
    
    # =============================================================================
    # API ROUTES
    # =============================================================================
    
    @app.route('/', methods=['GET'])
    def home():
        """API documentation page"""
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """System health check endpoint"""
        try:
            health_status = api_service.health_check()
            return jsonify({
                'status': 'success',
                'data': health_status
            }), 200
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/models', methods=['GET'])
    def get_models():
        """Get model information endpoint"""
        try:
            model_info = api_service.get_model_info()
            return jsonify({
                'status': 'success',
                'data': model_info
            }), 200
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Single prediction endpoint"""
        try:
            # Parse request data
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No JSON data provided'
                }), 400
            
            sensor_data = data.get('sensor_data')
            model_name = data.get('model_name', 'Random Forest')
            
            if not sensor_data:
                return jsonify({
                    'status': 'error',
                    'message': 'sensor_data is required'
                }), 400
            
            if not isinstance(sensor_data, list):
                return jsonify({
                    'status': 'error',
                    'message': 'sensor_data must be a list of numbers'
                }), 400
            
            # Make prediction
            result = api_service.predict_failure(sensor_data, model_name)
            
            return jsonify({
                'status': 'success',
                'data': result
            }), 200
            
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Prediction failed: {str(e)}'
            }), 500
    
    @app.route('/batch_predict', methods=['POST'])
    def batch_predict():
        """Batch prediction endpoint"""
        try:
            # Parse request data
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No JSON data provided'
                }), 400
            
            samples = data.get('samples')
            model_name = data.get('model_name', 'Random Forest')
            
            if not samples:
                return jsonify({
                    'status': 'error',
                    'message': 'samples is required'
                }), 400
            
            if not isinstance(samples, list):
                return jsonify({
                    'status': 'error',
                    'message': 'samples must be a list of sensor data arrays'
                }), 400
            
            # Make predictions for all samples
            results = []
            for i, sample in enumerate(samples):
                try:
                    result = api_service.predict_failure(sample, model_name)
                    result['sample_index'] = i
                    results.append(result)
                except Exception as e:
                    results.append({
                        'sample_index': i,
                        'error': str(e)
                    })
            
            return jsonify({
                'status': 'success',
                'data': {
                    'predictions': results,
                    'total_samples': len(samples),
                    'successful_predictions': len([r for r in results if 'error' not in r])
                }
            }), 200
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Batch prediction failed: {str(e)}'
            }), 500
    
    @app.route('/retrain', methods=['POST'])
    def retrain_models():
        """Retrain models with new data (if provided)"""
        try:
            data = request.get_json()
            
            # This is a placeholder - in production, you'd implement actual retraining logic
            return jsonify({
                'status': 'success',
                'message': 'Model retraining initiated',
                'data': {
                    'training_started': datetime.now().isoformat(),
                    'estimated_completion': 'To be implemented'
                }
            }), 202  # Accepted for processing
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Retraining failed: {str(e)}'
            }), 500
    
    # =============================================================================
    # ERROR HANDLERS
    # =============================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify({
            'status': 'error',
            'message': 'Endpoint not found'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify({
            'status': 'error',
            'message': 'Method not allowed'
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_sample_request():
    """Create a sample request for testing"""
    return {
        'sensor_data': np.random.normal(0, 1, 590).tolist(),
        'model_name': 'Random Forest'
    }

def test_api_locally():
    """Test the API locally without starting the server"""
    print("🧪 Testing API Service Locally")
    print("=" * 40)
    
    # Initialize API service
    api = PredictiveMaintenanceAPI()
    
    if not api.is_loaded:
        print("❌ No models loaded. Please train and save models first.")
        return False
    
    # Test model info
    print("1. Testing model info...")
    model_info = api.get_model_info()
    print(f"   Models loaded: {model_info['models_loaded']}")
    print(f"   Available models: {model_info['available_models']}")
    
    # Test health check
    print("\n2. Testing health check...")
    health = api.health_check()
    print(f"   API status: {health['api_status']}")
    print(f"   Models loaded: {health['models_loaded']}")
    
    # Test prediction
    print("\n3. Testing prediction...")
    sample_data = np.random.normal(0, 1, 590).tolist()
    
    try:
        result = api.predict_failure(sample_data, 'Random Forest')
        print(f"   Prediction: {result['prediction_label']}")
        print(f"   Alert level: {result['alert_level']}")
        if result['confidence']:
            print(f"   Confidence: {result['confidence']:.3f}")
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

def main():
    """Main function for running the API service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictive Maintenance API Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test', action='store_true', help='Run local tests instead of starting server')
    parser.add_argument('--models-dir', default='./models', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    if not FLASK_AVAILABLE:
        print("❌ Flask is not available. Install with: pip install flask flask-cors")
        return
    
    if args.test:
        # Run local tests
        success = test_api_locally()
        sys.exit(0 if success else 1)
    
    # Update model directory
    global api_service
    api_service = PredictiveMaintenanceAPI(args.models_dir)
    
    if not api_service.is_loaded:
        print("⚠️  Warning: No models loaded. API will have limited functionality.")
        print("   Please ensure models are trained and saved in the models directory.")
    
    print(f"🚀 Starting Predictive Maintenance API Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Debug: {args.debug}")
    print(f"   Models directory: {args.models_dir}")
    print(f"   Models loaded: {api_service.is_loaded}")
    print()
    print(f"📖 API Documentation: http://{args.host}:{args.port}/")
    print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
    print()
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n👋 API Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    main()
