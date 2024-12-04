# Use the official Python base iamge 
FROM python:3.9-slim

# Set the working directory inside the container 
WORKDIR /app

#Copy requirements file in the container
COPY requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code to the container 
COPY . .

# Expose the port FASTAPI will run on
EXPOSE 8000

#Run the FASTAPI app with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$PORT"]
