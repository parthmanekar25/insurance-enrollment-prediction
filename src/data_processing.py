"""
Data processing pipeline for insurance enrollment prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class DataProcessor:
    """Handles data loading, cleaning, and feature engineering."""
    
    def __init__(self):
        """Initialize the data processor with scaler and label encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw employee data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with raw employee data
        """
        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Drop employee_id as it's not a feature
        if 'employee_id' in df.columns:
            df = df.drop('employee_id', axis=1)
        
        # Handle missing values (fill with median for numerical, mode for categorical)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if col != 'enrolled':
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        Based on EDA analysis, focus on significant predictors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # 🔥 Salary per tenure year (avoid division by zero)
        # Captures earning efficiency
        df['salary_per_tenure'] = df['salary'] / (df['tenure_years'] + 1)
        
        # 🔥 Age groups (age is significant predictor: p=0.0000, corr=0.269)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 30, 45, 60, 100], 
                labels=['young', 'mid', 'senior', 'veteran']
            )
        
        # 🔥 High earner indicator (salary is significant: p=0.0000, corr=0.366)
        if 'salary' in df.columns:
            df['high_earner'] = (df['salary'] > df['salary'].quantile(0.75)).astype(int)
        
        # 🔥 High earner + Senior interaction (combine top 2 predictors)
        # Both age and salary are significant - interaction term captures their combined effect
        if 'age' in df.columns and 'salary' in df.columns:
            df['high_earner_and_senior'] = (
                (df['salary'] > df['salary'].quantile(0.75)) & 
                (df['age'] > df['age'].median())
            ).astype(int)
        
        # 🔥 Dependents binary flag (highest predictor: corr=0.453)
        if 'has_dependents' in df.columns:
            df['has_dependents_binary'] = (df['has_dependents'] == 'Yes').astype(int)
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding.
        
        Based on EDA analysis:
        - Keep: employment_type (p=0.0000), has_dependents (corr=0.453)
        - Drop: gender (p=0.5887), region (p=0.6147), marital_status (p=0.1942)
        
        Args:
            df: Input DataFrame
            fit: Whether to fit new encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Only encode significant categorical features + engineered categorical features
        significant_categorical_cols = ['employment_type', 'has_dependents', 'age_group']
        
        # Drop non-significant features to reduce noise
        cols_to_drop = ['gender', 'region', 'marital_status']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        for col in significant_categorical_cols:
            if col not in df.columns:
                continue
            
            # Skip target variable
            if col == 'enrolled':
                continue
            
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def prepare_train_test(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Split data into train/test sets and apply scaling.
        
        Based on EDA analysis, tenure_years is NOT significant (p=0.4545)
        and has weak negative correlation (-0.007), so it's excluded.
        
        Args:
            df: Input DataFrame with all preprocessing applied
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Drop non-significant features and ID
        cols_to_drop = ['enrolled', 'employee_id', 'tenure_years']
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = df['enrolled']
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Identify numerical columns for scaling
        num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        # Fit scaler on training data and transform both sets
        X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the scaler and label encoders for inference.
        
        Args:
            filepath: Path to save the preprocessor object
        """
        preprocessor = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        joblib.dump(preprocessor, filepath)
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load the scaler and label encoders for inference.
        
        Args:
            filepath: Path to the saved preprocessor object
        """
        preprocessor = joblib.load(filepath)
        self.scaler = preprocessor['scaler']
        self.label_encoders = preprocessor['label_encoders']


if __name__ == "__main__":
    # Test the processor
    processor = DataProcessor()
    try:
        df = processor.load_data('data/raw/employee_data.csv')
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print("Data file not found. Please ensure employee_data.csv exists in data/raw/")
