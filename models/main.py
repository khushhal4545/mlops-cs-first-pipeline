
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Create FastAPI application
app = FastAPI(title="Iris Model API")

# 2. Load the trained model
# Works when main.py is inside the models folder
model_path = Path(__file__).resolve().parent / "iris_model.joblib"
model = joblib.load(model_path)

# 3. Define input data format
class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# 4. Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# 5. Prediction endpoint
@app.post("/predict")
def predict_iris(request: IrisRequest):
    # Convert input into the format expected by the model
    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]

    # Predict the flower class
    prediction = int(model.predict(features)[0])

    # Map class number to species name
    species = ["setosa", "versicolor", "virginica"]
    class_name = species[prediction]

    # Return prediction details
    return {
        "prediction": prediction,
        "predicted_class": prediction,
        "class_name": class_name,
        "predicted_species": class_name
    }
