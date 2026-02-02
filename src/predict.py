"""
Prediction utilities for insurance enrollment prediction.
"""
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_processing import DataProcessor


class PredictionEngine:
    """Handles model predictions and preprocessing for new data."""
    
    def __init__(self, model_path: str = 'models/best_model.pkl',
                 preprocessor_path: str = 'models/preprocessor.pkl'):
        """
        Initialize prediction engine with pre-trained models.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model = joblib.load(model_path)
        self.processor = DataProcessor()
        self.processor.load_preprocessor(preprocessor_path)
        self.feature_names = None
    
    def preprocess_single(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess a single employee record for prediction.
        Applies the same feature engineering as training pipeline.
        
        Args:
            data: Dictionary with employee attributes (age, salary, employment_type, has_dependents)
            
        Returns:
            Processed DataFrame ready for model prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add tenure_years for feature engineering (assume new employee)
        df['tenure_years'] = 0.0
        
        # Apply same feature engineering as training
        # Salary per tenure year (avoid division by zero)
        df['salary_per_tenure'] = df['salary'] / (df['tenure_years'] + 1)
        
        # Age groups (age is significant predictor: p=0.0000, corr=0.269)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 30, 45, 60, 100],
                labels=['young', 'mid', 'senior', 'veteran']
            )
        
        # High earner indicator (salary is significant: p=0.0000, corr=0.366)
        # Use 75th percentile from training data (~$85,000)
        if 'salary' in df.columns:
            df['high_earner'] = (df['salary'] > df['salary'].quantile(0.75)).astype(int)
        
        # High earner + Senior interaction (combine top 2 predictors)
        if 'age' in df.columns and 'salary' in df.columns:
            df['high_earner_and_senior'] = (
                (df['salary'] > df['salary'].quantile(0.75)) & 
                (df['age'] > df['age'].median())
            ).astype(int)
        
        # Dependents binary flag (highest predictor: corr=0.453)
        if 'has_dependents' in df.columns:
            df['has_dependents_binary'] = (df['has_dependents'] == 'Yes').astype(int)
        
        # Drop temporary feature before encoding
        df = df.drop('tenure_years', axis=1, errors='ignore')
        
        # Encode categorical features
        df = self.processor.encode_features(df, fit=False)
        
        # Scale numerical features
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[num_cols] = self.processor.scaler.transform(df[num_cols])
        
        # Ensure features are in the same order as training
        # Expected order: age, salary, employment_type, has_dependents, salary_per_tenure, 
        #                age_group, high_earner, high_earner_and_senior, has_dependents_binary
        expected_features = ['age', 'salary', 'employment_type', 'has_dependents', 
                            'salary_per_tenure', 'age_group', 'high_earner', 
                            'high_earner_and_senior', 'has_dependents_binary']
        
        # Select only expected features (and in correct order)
        available_features = [f for f in expected_features if f in df.columns]
        df = df[available_features]
        
        return df
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single employee.
        
        Args:
            data: Dictionary with employee attributes
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Preprocess
        df = self.preprocess_single(data)
        
        # Predict
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        # Determine confidence level
        if probability >= 0.7 or probability <= 0.3:
            confidence = "high"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            'enrolled_probability': round(float(probability), 4),
            'prediction': int(prediction),
            'confidence': confidence
        }
    
    def batch_predict(self, data_list: list) -> pd.DataFrame:
        """
        Make predictions for multiple employees.
        
        Args:
            data_list: List of dictionaries with employee attributes
            
        Returns:
            DataFrame with predictions and probabilities
        """
        predictions = []
        
        for data in data_list:
            result = self.predict(data)
            result['input'] = data
            predictions.append(result)
        
        return pd.DataFrame(predictions)


def main():
    """Test prediction engine."""
    print("\n" + "="*60)
    print("PREDICTION ENGINE TEST")
    print("="*60 + "\n")
    
    try:
        # Initialize engine
        engine = PredictionEngine()
        print("✓ Prediction engine initialized\n")
        
        # Test prediction
        test_employee = {
            'age': 35,
            'salary': 75000,
            'employment_type': 'Full-time',
            'has_dependents': 'Yes'
        }
        
        print("Making prediction for test employee...")
        result = engine.predict(test_employee)
        
        print(f"\nPrediction Result:")
        print(f"  Enrollment Probability: {result['enrolled_probability']:.2%}")
        print(f"  Prediction: {'Enrolled' if result['prediction'] == 1 else 'Not Enrolled'}")
        print(f"  Confidence: {result['confidence'].upper()}")
        print("\n✓ Prediction complete!\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure models are trained first by running: python src/train.py")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
