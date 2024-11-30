from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_data
from src.model import train_model

# Initialize the FastAPI app
app = FastAPI()

# Define global variables
MODEL_PATH = "./optimized_model.h5"
UPLOAD_FOLDER = "data/uploaded_data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the existing model
model = load_model(MODEL_PATH)

@app.post("/predict/")
async def predict(features: list[float]):
    """
    Predict using the existing model given a list of feature values.

    Parameters:
        features (list): A list of feature values.

    Returns:
        dict: Prediction result.
    """
    try:
        # Convert features to numpy array and reshape for model
        input_data = np.array(features).reshape(1, len(features), 1)
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = int(np.round(prediction[0][0]))  # Round to 0 or 1

        return JSONResponse(content={
            "prediction": predicted_class,
            "probability": float(prediction[0][0])
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a new dataset for retraining the model.

    Parameters:
        file (UploadFile): The uploaded file.

    Returns:
        dict: Acknowledgment of file upload.
    """
    try:
        # Save uploaded file
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        return {"message": f"File '{file.filename}' uploaded successfully!"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/retrain/")
async def retrain_model(file_name: str = Form(...)):
    """
    Retrain the model using the uploaded dataset.

    Parameters:
        file_name (str): Name of the uploaded file.

    Returns:
        dict: Retraining results.
    """
    try:
        # Load the uploaded dataset
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": f"File '{file_name}' not found."}, status_code=404)

        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocess_data(file_path)

        # Retrain the model
        _, retrained_model = train_model(X_train, y_train, X_test, y_test)

        # Save the new model
        retrained_model.save(MODEL_PATH)

        return {"message": f"Model retrained successfully with '{file_name}'!"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
