# AI Text Classification Model

## Loading the Gradient Boost Model

The model is saved using **Joblib**. You can load it as follows:

```python
import joblib

# Load the model
model = joblib.load("export/gradient_boost_model.pkl")
```

## Making Predictions

Use the method below to classify text inputs:

```python
content = "Ubud, located in the heart of Bali, Indonesia, is a cultural and artistic haven renowned for its lush landscapes, vibrant traditions, and serene atmosphere."

result = model.predict_text(content)
print(result)
```

### Expected Output

```json
{
    "Prediction": "Human",
    "AI Probability": "12.34%"
}
```

## Notes

- Ensure your text input is in **string format**.
- The **threshold for classification** is based on probability (if >50%, it's AI-generated).
