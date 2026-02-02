# Insurance Enrollment Prediction

A production-ready machine learning pipeline to predict employee insurance enrollment likelihood with a FastAPI REST API for deployment.

![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Project Overview

This project implements an end-to-end machine learning solution to predict whether employees will enroll in insurance plans. It features:

- **Modular Architecture**: Clean separation of concerns (data, training, evaluation, API)
- **Multiple Models**: Logistic Regression baseline and XGBoost with automatic model selection
- **FastAPI Integration**: Production-ready REST API with interactive documentation
- **Comprehensive Evaluation**: Confusion matrix, ROC curves, feature importance visualization
- **Scalable Design**: Support for batch predictions and multi-threaded deployment

## 📁 Project Structure

```
insurance-enrollment-ml/
├── data/
│   ├── raw/
│   │   └── employee_data.csv              # Raw employee data
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py                 # Data cleaning & feature engineering
│   ├── train.py                           # Model training pipeline
│   ├── evaluate.py                        # Model evaluation & visualization
│   └── predict.py                         # Prediction utilities & inference
├── api/
│   ├── __init__.py
│   ├── main.py                            # FastAPI application
│   └── schemas.py                         # Pydantic request/response models
├── models/
│   ├── best_model.pkl                     # Trained model
│   └── preprocessor.pkl                   # Scaler & encoders
├── notebooks/
│   └── eda.ipynb                          # Exploratory Data Analysis
├── tests/
│   └── test_api.py                        # API endpoint tests
├── requirements.txt                       # Project dependencies
├── README.md                              # This file
├── report.md                              # Technical report
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Installation

1. **Clone or navigate to the project directory**
```bash
cd insurance-enrollment-ml
```

2. **Create and activate virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n insurance-ml python=3.9
conda activate insurance-ml
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Training the Model

1. **Prepare your data**
   - Place your `employee_data.csv` in the `data/raw/` directory
   - Required columns: age, gender, marital_status, salary, employment_type, region, has_dependents, tenure_years, enrolled

2. **Run the training pipeline**
```bash
python src/train.py
```

This will:
- Load and clean the data
- Engineer features
- Train both Logistic Regression and XGBoost models
- Select the best model based on cross-validation ROC-AUC
- Save the model and preprocessor to `models/`

### Running the API

1. **Start the FastAPI server**
```bash
cd api
uvicorn main:app --reload --port 8000
```

Or from the project root:
```bash
python -m uvicorn api.main:app --reload --port 8000
```

2. **Access the API**
   - Interactive API docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - API endpoint: http://localhost:8000/

### Making Predictions

#### Single Prediction with cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "salary": 75000,
    "employment_type": "Full-time",
    "region": "West",
    "has_dependents": 1,
    "tenure_years": 5.5
  }'
```

#### Response
```json
{
  "enrolled_probability": 0.7542,
  "prediction": 1,
  "confidence": "high"
}
```

#### Batch Predictions with Python
```python
import requests

url = "http://localhost:8000/batch-predict"
employees = [
    {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "salary": 75000,
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": 1,
        "tenure_years": 5.5
    },
    # ... more employees
]

response = requests.post(url, json=employees)
predictions = response.json()["predictions"]
```

### Model Evaluation

Run the evaluation script to generate performance metrics and visualizations:

```bash
python src/evaluate.py
```

This generates:
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC-AUC curve
- `feature_importance.png` - Top 15 important features

### Testing

Run API tests:
```bash
python -m pytest tests/test_api.py -v
```

Or using the test script:
```bash
python tests/test_api.py
```

## 📊 Data Requirements

### Input Features

| Feature | Type | Description | Constraints |
|---------|------|-------------|-------------|
| age | integer | Employee age | 18-100 |
| gender | string | Gender | Categorical |
| marital_status | string | Marital status | Categorical |
| salary | float | Annual salary (USD) | > 0 |
| employment_type | string | Type of employment | Categorical |
| region | string | Geographic region | Categorical |
| has_dependents | integer | Has dependents | 0 or 1 |
| tenure_years | float | Years of service | ≥ 0 |
| enrolled | integer | Target variable | 0 or 1 (training only) |

### Data Format

CSV file with headers and one record per row:

```csv
age,gender,marital_status,salary,employment_type,region,has_dependents,tenure_years,enrolled
35,Male,Married,75000,Full-time,West,1,5.5,1
42,Female,Single,85000,Full-time,East,0,10.0,1
28,Male,Single,55000,Part-time,South,0,2.0,0
```

## 🔧 Development

### Project Configuration

Key parameters can be adjusted in the source code:

**Data Processing** (`src/data_processing.py`):
- Test/train split ratio
- Scaler type
- Feature engineering thresholds

**Model Training** (`src/train.py`):
- XGBoost hyperparameters
- Cross-validation folds
- Model selection criteria

**API** (`api/main.py`):
- Batch size limits
- CORS settings
- Port and host configuration

### Code Style

- Follow PEP 8 conventions
- Type hints for function parameters
- Docstrings for all classes and functions
- Modular design with single responsibility principle

## 📈 Performance Metrics

After training, you'll see metrics like:

- **ROC-AUC**: 0.8234 (example)
- **Precision**: 0.82
- **Recall**: 0.79
- **F1-Score**: 0.80

See `report.md` for detailed results.

## 🔐 API Security

For production deployment:

1. **Remove CORS wildcard**
```python
allow_origins=["https://yourdomain.com"]  # Specify allowed origins
```

2. **Add authentication**
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
async def predict_enrollment(employee: EmployeeData, credentials: HTTPAuthCredentials = Depends(security)):
    # Validate credentials
    ...
```

3. **Add rate limiting**
```bash
pip install slowapi
```

4. **Use HTTPS in production**
```bash
uvicorn api.main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.0 | ML models & preprocessing |
| xgboost | 2.0.0 | Gradient boosting |
| fastapi | 0.104.1 | Web API framework |
| uvicorn | 0.24.0 | ASGI server |
| pydantic | 2.5.0 | Data validation |
| joblib | 1.3.2 | Model serialization |
| matplotlib | 3.7.2 | Visualizations |
| seaborn | 0.12.2 | Statistical plots |

## 🚢 Deployment

### Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t insurance-ml .
docker run -p 8000:8000 insurance-ml
```

### Cloud Deployment

**AWS EC2**:
```bash
# SSH into instance
ssh -i key.pem ubuntu@instance-ip

# Clone repo, install, and run
git clone <repo>
cd insurance-enrollment-ml
pip install -r requirements.txt
python src/train.py
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Heroku**:
```bash
heroku create insurance-enrollment-ml
git push heroku main
```

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Last Updated**: February 2026
**Maintainer**: Your Name
