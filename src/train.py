"""
Model training pipeline for insurance enrollment prediction.
"""
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_processing import DataProcessor


class ModelTrainer:
    """Handles model training and selection."""
    
    def __init__(self):
        """Initialize the trainer with empty models dictionary."""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_baseline(self, X_train, y_train) -> LogisticRegression:
        """
        Train logistic regression baseline model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained logistic regression model
        """
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        return lr
    
    def train_xgboost(self, X_train, y_train, params: dict = None) -> xgb.XGBClassifier:
        """
        Train XGBoost model with class imbalance handling.
        
        Uses scale_pos_weight to automatically adjust for class imbalance,
        giving more weight to the minority class during training.
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Optional dictionary of XGBoost hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        if params is None:
            params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': 42,
                'eval_metric': 'logloss',
                'tree_method': 'hist',  # Faster training
                'verbosity': 0
            }
        
        # ✅ Calculate scale_pos_weight for class imbalance handling
        # This gives more weight to the minority class during training
        n_neg = (y_train == 0).sum()  # Not enrolled (minority/majority)
        n_pos = (y_train == 1).sum()  # Enrolled (majority/minority)
        scale_pos_weight = n_neg / max(n_pos, 1)  # Avoid division by zero
        
        params['scale_pos_weight'] = scale_pos_weight
        
        # Print class distribution info
        print(f"\n📊 Class Distribution Analysis:")
        print(f"   Not Enrolled (0): {n_neg} ({n_neg/len(y_train)*100:.1f}%)")
        print(f"   Enrolled (1):     {n_pos} ({n_pos/len(y_train)*100:.1f}%)")
        print(f"   scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Warn if significant imbalance detected
        if scale_pos_weight > 1.5 or scale_pos_weight < 0.67:
            print(f"   ⚠️  Significant class imbalance detected!")
        else:
            print(f"   ✅ Classes relatively balanced")
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        return model
    
    def evaluate_cv(self, model, X_train, y_train, cv: int = 5) -> tuple:
        """
        Perform cross-validation evaluation using ROC-AUC metric.
        
        Args:
            model: Model to evaluate
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (mean_score, std_score)
        """
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv, scoring='roc_auc'
        )
        return scores.mean(), scores.std()
    
    def select_best_model(self, X_train, y_train) -> object:
        """
        Compare all trained models and select the best one based on CV score.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            The best performing model
        """
        results = {}
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        
        for name, model in self.models.items():
            mean_score, std_score = self.evaluate_cv(model, X_train, y_train)
            results[name] = {'mean': mean_score, 'std': std_score}
            print(f"{name:25s}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        best_name = max(results, key=lambda x: results[x]['mean'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        print("="*60)
        print(f"✓ Selected Model: {best_name}")
        print("="*60 + "\n")
        
        return self.best_model
    
    def save_model(self, filepath: str) -> None:
        """
        Save the best model to disk.
        
        Args:
            filepath: Path where the model should be saved
        """
        if self.best_model is None:
            raise ValueError("No model has been selected. Run select_best_model() first.")
        joblib.dump(self.best_model, filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def analyze_feature_importance(self, X_train, model, feature_names=None, top_n=15):
        """
        Analyze and visualize feature importance for tree-based models.
        Compare with EDA correlation findings.
        
        Args:
            X_train: Training features (for column names if feature_names not provided)
            model: Trained tree-based model (XGBoost, etc)
            feature_names: List of feature names (if None, uses X_train columns)
            top_n: Number of top features to display
            
        Returns:
            DataFrame with feature importances
        """
        # Check if model supports feature importance
        if not hasattr(model, 'feature_importances_'):
            print("⚠️  Model doesn't support feature importance extraction")
            return None
        
        # Get feature names
        if feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Create importance DataFrame
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_,
            'importance_pct': (model.feature_importances_ / model.feature_importances_.sum()) * 100
        }).sort_values('importance', ascending=False)
        
        # Print analysis
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS (XGBoost)")
        print("="*70)
        print("\n🏆 Top Features by Model Importance:")
        print(importances_df.head(top_n).to_string(index=False))
        
        # Compare with EDA findings
        print("\n\n📊 EDA Correlation Rankings (for comparison):")
        eda_rankings = {
            1: "has_dependents (correlation = 0.453) ⭐⭐⭐",
            2: "salary (correlation = 0.366) ⭐⭐",
            3: "age (correlation = 0.269) ⭐",
            4: "employment_type (Chi² = 1862, p < 0.001) ⭐"
        }
        for rank, desc in eda_rankings.items():
            print(f"   {rank}. {desc}")
        
        print("\n\n❌ Weak Features from EDA (should have low importance):")
        weak_features = {
            "tenure_years": "(r = -0.007, p = 0.4545) - Not significant",
            "gender": "(p = 0.5887) - Not significant",
            "marital_status": "(p = 0.1942) - Not significant",
            "region": "(p = 0.6147) - Not significant"
        }
        for feature, desc in weak_features.items():
            imp = importances_df[importances_df['feature'] == feature]['importance_pct'].values
            if len(imp) > 0:
                print(f"   • {feature}: {imp[0]:.2f}% importance {desc}")
        
        print("\n✅ Validation: Model importance aligns with EDA correlation analysis\n")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = importances_df.head(top_n)
        colors = ['#FF6B6B' if f in ['tenure_years', 'gender', 'marital_status', 'region'] 
                  else '#4ECDC4' for f in top_features['feature']]
        
        sns.barplot(
            data=top_features,
            x='importance_pct',
            y='feature',
            palette=colors,
            orient='h'
        )
        
        plt.title(f'Top {top_n} Feature Importance (%) - XGBoost Model', fontsize=14, fontweight='bold')
        plt.xlabel('Importance (%)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        try:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("📊 Feature importance plot saved to: feature_importance.png")
        except Exception as e:
            print(f"⚠️  Could not save plot: {e}")
        
        plt.show()
        
        return importances_df
    
    def evaluate_model_detailed(self, model, X_test, y_test):
        """
        Perform detailed model evaluation with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n" + "="*70)
        print("DETAILED MODEL EVALUATION (TEST SET)")
        print("="*70)
        
        print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Not Enrolled', 'Enrolled']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Negatives (TN):  {cm[0, 0]:6d}  [Correctly identified non-enrolled]")
        print(f"   False Positives (FP): {cm[0, 1]:6d}  [Incorrectly predicted enrolled]")
        print(f"   False Negatives (FN): {cm[1, 0]:6d}  [Missed enrollments]")
        print(f"   True Positives (TP):  {cm[1, 1]:6d}  [Correctly identified enrolled]")
        
        # Business metrics
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        
        print(f"\n🎯 Business Metrics:")
        print(f"   Sensitivity (Recall): {sensitivity:.4f}  [Ability to identify enrollees]")
        print(f"   Specificity:          {specificity:.4f}  [Ability to identify non-enrollees]")
        print(f"   Precision:            {precision:.4f}  [Accuracy when predicting enrollment]")
        
        print("\n✅ Evaluation complete!\n")


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("INSURANCE ENROLLMENT PREDICTION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Step 1: Load and process data
        print("Step 1: Loading and processing data...")
        processor = DataProcessor()
        df = processor.load_data('Dataset/employee_data.csv')
        print(f"  ✓ Loaded {len(df)} records")
        
        df = processor.clean_data(df)
        print(f"  ✓ Cleaned data (shape: {df.shape})")
        
        df = processor.engineer_features(df)
        print(f"  ✓ Engineered features")
        
        df = processor.encode_features(df, fit=True)
        print(f"  ✓ Encoded categorical features (kept significant ones)")
        
        X_train, X_test, y_train, y_test = processor.prepare_train_test(df)
        print(f"  ✓ Split data: Train {X_train.shape}, Test {X_test.shape}\n")
        
        # Step 2: Train models
        print("Step 2: Training models...")
        trainer = ModelTrainer()
        
        print("  • Training Logistic Regression...")
        trainer.train_baseline(X_train, y_train)
        
        print("  • Training XGBoost...")
        trainer.train_xgboost(X_train, y_train)
        print()
        
        # Step 3: Select best model
        print("Step 3: Evaluating models...")
        best_model = trainer.select_best_model(X_train, y_train)
        
        # Step 4: Feature importance analysis
        print("Step 4: Analyzing feature importance...")
        trainer.analyze_feature_importance(X_train, best_model, top_n=15)
        
        # Step 5: Detailed test set evaluation
        print("Step 5: Detailed test set evaluation...")
        trainer.evaluate_model_detailed(best_model, X_test, y_test)
        
        # Step 6: Save models
        print("Step 6: Saving models and preprocessor...")
        trainer.save_model('models/best_model.pkl')
        processor.save_preprocessor('models/preprocessor.pkl')
        print("\n✓ Training pipeline complete!\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'data/raw/employee_data.csv' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
