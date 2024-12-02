from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware


# Initialize the FastAPI app
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Path to the saved model
MODEL_PATH = "models/best_model.pkl"

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {str(e)}")

# Define Pydantic model for input validation
class DiabetesFeatures(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, description="2-hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, description="Body mass index (kg/m^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: int = Field(..., ge=0, description="Age in years")

@app.post("/predict/")
async def predict(data: DiabetesFeatures):
    """
    Predict the outcome (diabetes or not) based on clinical features.
    
    Parameters:
        data (DiabetesFeatures): Input features validated by Pydantic.
    
    Returns:
        dict: Predicted outcome and probability.
    """
    try:
        # Convert the input data to a NumPy array
        input_features = np.array([
            [
                data.Pregnancies,
                data.Glucose,
                data.BloodPressure,
                data.SkinThickness,
                data.Insulin,
                data.BMI,
                data.DiabetesPedigreeFunction,
                data.Age,
            ]
        ])
        
        # Make prediction
        prediction = model.predict(input_features)
        probability = model.predict_proba(input_features)[:, 1]  # Probability for class 1

        # Prepare the response
        response = {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Example Endpoint to Check API Status
@app.get("/")
async def read_root():
    return {"message": "API is up and running!"}

