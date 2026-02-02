"""
Model evaluation and reporting for insurance enrollment prediction.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_processing import DataProcessor


def evaluate_model(model, X_test, y_test, output_dir: str = '.'):
    """
    Comprehensive model evaluation with visualizations and metrics.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test target
        output_dir: Directory to save evaluation plots
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Enrolled', 'Enrolled']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualizations
    plot_confusion_matrix(cm, output_dir)
    plot_roc_curve(y_test, y_pred_proba, output_dir)
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, X_test, output_dir)
    
    print("="*60 + "\n")
    
    return {
        'auc': auc_score,
        'f1': f1,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, output_dir: str = '.'):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Enrolled', 'Enrolled'],
                yticklabels=['Not Enrolled', 'Enrolled'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved to: {output_dir}/confusion_matrix.png")


def plot_roc_curve(y_test, y_pred_proba, output_dir: str = '.'):
    """
    Plot and save ROC curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save the plot
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curve saved to: {output_dir}/roc_curve.png")


def plot_feature_importance(model, X_test, output_dir: str = '.', top_n: int = 15):
    """
    Plot and save top feature importances.
    
    Args:
        model: Model with feature_importances_ attribute
        X_test: Test features (to get column names)
        output_dir: Directory to save the plot
        top_n: Number of top features to display
    """
    importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    top_features = importances.head(top_n)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance plot saved to: {output_dir}/feature_importance.png")
    
    print(f"\nTop 10 Features:")
    for idx, row in importances.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("INSURANCE ENROLLMENT PREDICTION - EVALUATION")
    print("="*60 + "\n")
    
    try:
        # Load model and preprocessor
        print("Loading trained model and preprocessor...")
        model = joblib.load('models/best_model.pkl')
        processor = DataProcessor()
        processor.load_preprocessor('models/preprocessor.pkl')
        print("✓ Model and preprocessor loaded\n")
        
        # Load test data
        print("Loading and processing test data...")
        df = processor.load_data('data/raw/employee_data.csv')
        df = processor.clean_data(df)
        df = processor.engineer_features(df)
        df = processor.encode_features(df, fit=False)
        
        X_train, X_test, y_train, y_test = processor.prepare_train_test(df)
        print(f"✓ Test set shape: {X_test.shape}\n")
        
        # Evaluate
        print("Step 1: Generating evaluation metrics and plots...")
        evaluate_model(model, X_test, y_test, output_dir='.')
        print("Step 2: Evaluation complete!\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure models are trained first by running: python src/train.py")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
