# 🚀 API Quick Start Guide

## Starting the API Server

```bash
cd /Users/parth/Projects/Predicting\ Insurance\ Enrollment
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Making Predictions

### Single Prediction Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "salary": 75000,
    "employment_type": "Full-time",
    "has_dependents": "Yes"
  }'
```

**Response:**
```json
{
    "enrolled_probability": 1.0,
    "prediction": 1,
    "confidence": "high"
}
```

---

### Batch Prediction Example

```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 25, "salary": 30000, "employment_type": "Part-time", "has_dependents": "No"},
    {"age": 45, "salary": 75000, "employment_type": "Full-time", "has_dependents": "Yes"},
    {"age": 60, "salary": 100000, "employment_type": "Full-time", "has_dependents": "Yes"}
  ]'
```

---

## Documentation

### Interactive Swagger UI
Open in browser: **http://localhost:8000/docs**
- Test endpoints
- View request/response schemas
- See example values

### Alternative ReDoc View
Open in browser: **http://localhost:8000/redoc**
- Clean documentation layout
- Better for mobile viewing

---

## Input Field Requirements

| Field | Type | Valid Values | Example |
|-------|------|--------------|---------|
| `age` | Integer | 18 - 100 | 45 |
| `salary` | Number | > 0 | 75000 |
| `employment_type` | String | "Full-time", "Part-time", "Contract" | "Full-time" |
| `has_dependents` | String | "Yes", "No" | "Yes" |

---

## Understanding the Response

```json
{
    "enrolled_probability": 1.0,      // Probability between 0.0 and 1.0
    "prediction": 1,                  // 0 = Not Enrolled, 1 = Enrolled
    "confidence": "high"              // low, medium, or high
}
```

### Confidence Levels
- **high**: Prediction probability ≥ 0.70 or ≤ 0.30
- **medium**: Prediction probability 0.60-0.70 or 0.30-0.40
- **low**: Prediction probability 0.40-0.60

---

## API Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

---

## Error Examples

### Invalid Input
```bash
curl -X POST http://localhost:8000/predict \
  -d '{"age": "invalid", "salary": 50000, "employment_type": "Full-time", "has_dependents": "Yes"}'
```

**Response (400):**
```json
{
    "detail": [
        {
            "type": "value_error.number.not_a_number",
            "loc": ["body", "age"],
            "msg": "value is not a valid integer"
        }
    ]
}
```

### Model Not Ready
```bash
# Before model training
curl http://localhost:8000/predict
```

**Response (503):**
```json
{
    "detail": "Model is not loaded. Please train the model first."
}
```

---

## Python Integration

### Using Python Requests

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Single prediction
employee = {
    "age": 45,
    "salary": 75000,
    "employment_type": "Full-time",
    "has_dependents": "Yes"
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=employee,
    headers={"Content-Type": "application/json"}
)

result = response.json()
print(f"Enrolled: {result['prediction']}")
print(f"Probability: {result['enrolled_probability']:.1%}")
print(f"Confidence: {result['confidence']}")
```

### Batch Predictions

```python
employees = [
    {"age": 25, "salary": 30000, "employment_type": "Part-time", "has_dependents": "No"},
    {"age": 45, "salary": 75000, "employment_type": "Full-time", "has_dependents": "Yes"},
]

response = requests.post(
    f"{BASE_URL}/batch-predict",
    json=employees,
    headers={"Content-Type": "application/json"}
)

results = response.json()
print(f"Predictions for {results['count']} employees")
for pred in results['predictions']:
    print(f"  Age {pred['employee_data']['age']}: {pred['enrolled_probability']:.1%}")
```

---

## Stopping the Server

```bash
# Option 1: Ctrl+C in the terminal running the server
# Option 2: Use pkill
pkill -f "uvicorn api.main:app"
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Model Load Errors
1. Ensure `models/best_model.pkl` exists
2. Ensure `models/preprocessor.pkl` exists
3. Restart the server
4. Check OpenMP is installed: `ls /opt/homebrew/opt/libomp/lib/`

### Connection Refused
1. Verify server is running: `curl http://localhost:8000/health`
2. Check port isn't blocked by firewall
3. Restart server if needed

---

## Performance Tips

- **Batch Processing**: Use `/batch-predict` for 10+ predictions (more efficient)
- **Response Time**: ~50ms per single prediction
- **Max Batch Size**: 1000 employees per request
- **Concurrent Requests**: API handles multiple simultaneous requests

---

## Monitoring

### Check Server Logs
The server logs all incoming requests:
```
INFO:     127.0.0.1:53361 - "POST /predict HTTP/1.1" 200 OK
```

### Key Metrics to Monitor
- Response time (should be <100ms)
- Error rate (should be <1%)
- Model drift (recheck performance periodically)

---

## API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/info` | API information |
| POST | `/predict` | Single prediction |
| POST | `/batch-predict` | Batch predictions |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

---

## Next Steps

1. ✅ API is running on port 8000
2. ✅ All endpoints are tested and working
3. ✅ Interactive docs available at /docs
4. 📊 Monitor predictions in production
5. 🔄 Plan quarterly model retraining

Enjoy your predictions! 🎉

