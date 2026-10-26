# from pathlib import Path

# import joblib
# from fastapi import FastAPI
# from pydantic import BaseModel, Field


# # 1. Create FastAPI application
# app = FastAPI(title="Iris Model API")


# # 2. Load the trained model
# model_path = Path(__file__).resolve().parent / "iris_model.joblib"
# model = joblib.load(model_path)


# # 3. Define input data format
# class IrisRequest(BaseModel):
#     sepal_length: float = Field(gt=0)
#     sepal_width: float = Field(gt=0)
#     petal_length: float = Field(gt=0)
#     petal_width: float = Field(gt=0)


# # 4. Health check endpoint
# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}


# # 5. Prediction endpoint
# @app.post("/predict")
# def predict_iris(request: IrisRequest):

#     # Convert input into the format expected by the model
#     input_data = [[
#         request.sepal_length,
#         request.sepal_width,
#         request.petal_length,
#         request.petal_width
#     ]]

#     # Predict the flower class
#     prediction = int(model.predict(input_data)[0])

#     # Map class number to species name
#     species = ["setosa", "versicolor", "virginica"]

#     return {
#         "predicted_class": prediction,
#         "predicted_species": species[prediction]
#     }




from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Model API")

model_path = Path("models/iris_model.joblib")
model = joblib.load(model_path)

class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    
    return {
        "predicted_class": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }
    