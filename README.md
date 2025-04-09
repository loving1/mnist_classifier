# MNIST Digit Classifier

This project is a complete end-to-end application that allows users to draw digits and have them classified using a PyTorch model trained on the MNIST dataset.

## Components

1. **PyTorch Model**: A CNN model trained on the MNIST dataset for digit recognition.
2. **Streamlit Web App**: An interactive web interface where users can draw digits and get predictions.
3. **PostgreSQL Database**: Stores prediction logs including timestamp, predicted digit, confidence, and user feedback.

## How to Run

### Prerequisites

- Docker and Docker Compose

### Running the Application

1. Clone this repository
2. Navigate to the project directory
3. Run the application using Docker Compose:

```bash
docker-compose up -d
```

4. Access the web interface at `http://localhost:8501`

## Architecture

- **Model Training**: A PyTorch CNN model is trained on the MNIST dataset.
- **Web Interface**: Streamlit provides a canvas where users can draw digits.
- **Database**: PostgreSQL logs all predictions and user feedback.
- **Containerization**: All components are containerized using Docker for easy deployment.

## Live Demo

The application is available at: [http://your-server-ip:8501](http://your-server-ip:8501)

## Project Structure

```
mnist-classifier/
├── app/                 # Streamlit web application
│   ├── app.py           # Main Streamlit app
│   ├── Dockerfile       # Dockerfile for the web app
│   └── requirements.txt # App dependencies
├── model/               # Model training and inference
│   ├── train_model.py   # Script to train the PyTorch model
│   ├── predictor.py     # Interface for model inference
│   ├── Dockerfile       # Dockerfile for model training
│   └── requirements.txt # Model dependencies
├── database/            # Database scripts
│   └── init.sql         # SQL initialization script
└── docker-compose.yml   # Docker Compose configuration
```