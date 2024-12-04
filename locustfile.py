from locust import HttpUser, task, between
import random

class DiabetesPredictionUser(HttpUser):
    # Base host for the API
    host = "http://127.0.0.1:8000"  # Replace with your FastAPI URL
    wait_time = between(1, 5)

    @task
    def predict_diabetes(self):
        payload = {
            "Pregnancies": random.randint(0, 15),
            "Glucose": random.uniform(50.0, 200.0),
            "BloodPressure": random.uniform(40.0, 120.0),
            "SkinThickness": random.uniform(10.0, 50.0),
            "Insulin": random.uniform(0.0, 300.0),
            "BMI": random.uniform(18.0, 40.0),
            "DiabetesPedigreeFunction": random.uniform(0.1, 2.5),
            "Age": random.randint(20, 80),
        }
        with self.client.post("/predict", json=payload, catch_response=True, timeout=10) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")
            print(f"Payload: {payload}")
            print(f"Response: {response.status_code} - {response.text}")

    @task
    def check_api_status(self):
        with self.client.get("/", name="Check API Status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("API status check failed!")
