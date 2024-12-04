
# Diabetes Prediction API

A machine learning project that predicts the likelihood of diabetes based on user-provided clinical features such as glucose levels, BMI, and age. This project includes model training, testing, deployment, and stress testing for production readiness.

---

## 📽 Video Demo  
https://www.loom.com/share/d88a7ea41cda4a1ab5ffa61d5763a2a8?sid=c6f3813c-3d3e-4bac-ab77-d641a45ffab3 

---

## 🌐 Live Demo  
 API Endpoints https://ml-pipeline-summative.onrender.com/docs

---

## 📖 Project Description  

This project aims to predict whether a patient has diabetes or not based on medical features.  

### Key Features:  
1. Accepts user inputs via an API interface.  
2. Utilizes a trained machine learning model to predict diabetes likelihood.  
3. Includes model retraining functionality triggered by user-uploaded data.  

The backend is implemented using FastAPI, while the trained model is serialized as a `.pkl` file. The prediction model is trained on the **Pima Indian Diabetes dataset** using Scikit-learn.

---

## 🚀 Features  
- Machine Learning-based predictions for diabetes diagnosis.  
- Load and stress testing using **Locust**.  
- Model retraining triggered by user-uploaded datasets.  
- Dockerized for seamless deployment and scalability.  

---

## 🛠️ Setup Instructions  

### Prerequisites  
- Python 3.9 or higher  
- FastAPI  
- Scikit-learn  
- Docker (optional for containerized deployment)

### Steps to Set Up Locally  
1. **Clone the repository:**  
   https://github.com/StellaIbeh/ML-pipeline-Summative-

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend server:**  
   ```bash
   uvicorn api.main:app --reload
   ```

4. **Access the API documentation:**  
   Visit `http://127.0.0.1:8000/docs` to test the API endpoints.

---

## 📊 Flood Request Simulation Results  

The API was tested for performance under load using **Locust**.  
- **Maximum Requests:** 5000  
- **Success Rate:** 99%  
- **Average Response Time:** ~120ms  

Results demonstrate the application’s ability to handle high-traffic scenarios efficiently.

---

##  Notebook [Notebook](../../Downloads/Diabetes_Prediction_MLp_Summative.ipynb) 

The Jupyter Notebook includes all preprocessing, model training, and evaluation steps.  

### Key Sections:  
1. **Preprocessing Functions:** Handling missing values, scaling, and feature engineering.  
2. **Model Training:** Training with logistic regression and evaluating performance metrics like accuracy and F1 score.  
3. **Model Testing and Prediction:** Validating the model and generating predictions for new data.

---

## 🧠 Model File  

- **Trained Model:** Available as a `.pkl` file in the `models/` directory.  

To use the model:  
```python
import joblib
model = joblib.load('models/best_model.pkl')
prediction = model.predict(new_data)

```

---
# [bestmodels.pkl] (Users\HP\Downloads\best_model.pkl)

## 📦 Deployment Package  

### Option 1: Public URL (Render)  
The API is deployed live using Render for public access.  

### Option 2: Dockerized Application  
**Build and run the Docker image locally:**  
```bash
docker build -t diabetes-prediction-app
docker run -p 8000:8000 diabetes-prediction-app
```

---

## 📊 Visualizations  

Interpretations of the dataset features were created to provide insights:  
1.    ![feature selection](https://github.com/user-attachments/assets/a83a0cc7-5a63-4339-b384-d4cded2843ed)
Glucose level, BMI, and Age are strong predictors of diabetes.
2.  
3. ![image 2](<correlation matrix-1.png>) Highlights relationships between features.  
 

---

## 🤝 Contributing  

Contributions are welcome! Fork the repository and submit a pull request for review.

---

## 📄 License  

This project is licensed under the MIT License.

---

### 📧 Contact  

For inquiries or issues, contact:  
[Your Email](mailto:stellaibeh702@gmail.com)

---
