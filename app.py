import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model.",
    version="1.0.0"
)

MODEL_PATH = "churn_model.joblib"

# Load the currently selected model
model = joblib.load(MODEL_PATH)

# Define the required input fields
class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:
        # Convert the incoming customer data into a DataFrame
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        # Generate prediction
        prediction = int(
            model.predict(customer_df)[0]
        )

        # Calculate churn probability
        probabilities = model.predict_proba(
            customer_df
        )[0]

        churn_probability = float(probabilities[1])

        return {
            "churn_prediction": (
                "Yes" if prediction == 1 else "No"
            ),
            "churn_probability": round(
                churn_probability, 4
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(error)}"
        )