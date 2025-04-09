FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model training code
COPY model/ /app/model/

# Set working directory to model directory
WORKDIR /app/model

# Train model on container start if model doesn't exist
CMD ["python", "train_model.py"]