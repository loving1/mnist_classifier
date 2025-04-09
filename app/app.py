import streamlit as st
import numpy as np
import os
import sys
import psycopg2
from datetime import datetime
import json
from PIL import Image
import io
import base64

# Add the model directory to the path
sys.path.append(os.path.abspath("../model"))
from predictor import MNISTPredictor

# Initialize the model predictor
@st.cache_resource
def load_model():
    return MNISTPredictor(model_path="/app/model/mnist_model_scripted.pt")

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "database"),
        database=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres")
    )
    return conn

# Log prediction to database
def log_prediction(prediction, confidence, user_label=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                prediction INTEGER,
                confidence FLOAT,
                user_label INTEGER,
                probabilities JSON
            )
        """)
        
        # Insert record
        cursor.execute(
            "INSERT INTO predictions (timestamp, prediction, confidence, user_label, probabilities) VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(), prediction, confidence, user_label, json.dumps(st.session_state.get("probabilities", [])))
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# Initialize canvas drawing
def init_canvas():
    canvas_result = st.session_state.get("canvas_result", None)
    if canvas_result is None:
        st.session_state.canvas_result = np.zeros((280, 280))
    return st.session_state.canvas_result

# Streamlit app
def main():
    st.title("MNIST Digit Classifier")
    
    # Load the model
    predictor = load_model()
    
    # Initialize session state for canvas
    if "canvas_result" not in st.session_state:
        st.session_state.canvas_result = np.zeros((280, 280))
    
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    
    # Create drawing interface using HTML canvas
    st.markdown("""
    <style>
    .canvas-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 20px;
    }
    canvas {
        border: 2px solid #000;
        cursor: crosshair;
    }
    </style>
    
    <div class="canvas-container">
        <canvas id="drawing-board" width="280" height="280"></canvas>
        <div style="margin-top: 10px;">
            <button id="clear-button">Clear Canvas</button>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('drawing-board');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        // Set canvas background to black
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Set drawing settings
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';
        
        // Drawing functions
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }
        
        // Clear canvas
        document.getElementById('clear-button').addEventListener('click', () => {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        });
        
        // Event listeners
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        });
        
        canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup');
            canvas.dispatchEvent(mouseEvent);
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Add buttons for handling the prediction
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Predict Digit"):
            # Get the canvas image data
            canvas_data = st.session_state.get("canvas_data")
            if canvas_data:
                # Process the image
                try:
                    # Decode the base64 image
                    img_bytes = base64.b64decode(canvas_data.split(',')[1])
                    img = Image.open(io.BytesIO(img_bytes)).convert('L')
                    img = img.resize((28, 28))
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img) / 255.0
                    
                    # Make prediction
                    result = predictor.predict_from_array(img_array)
                    
                    # Store results in session state
                    st.session_state.prediction = result["prediction"]
                    st.session_state.confidence = result["confidence"]
                    st.session_state.probabilities = result["probabilities"]
                    st.session_state.prediction_made = True
                    
                    # Display results
                    st.success(f"Prediction: {result['prediction']}")
                    st.info(f"Confidence: {result['confidence']:.2%}")
                    
                    # Log prediction without user label (will be updated later)
                    log_prediction(result["prediction"], result["confidence"])
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
            else:
                st.warning("Please draw a digit first!")
    
    with col2:
        if st.button("Clear Canvas"):
            st.session_state.canvas_result = np.zeros((280, 280))
            st.session_state.prediction_made = False
            st.experimental_rerun()
    
    # Add component to capture canvas data from JavaScript
    components.html(
        """
        <script>
            // Function to send canvas data to Streamlit
            function sendCanvasData() {
                const canvas = document.getElementById('drawing-board');
                const imageData = canvas.toDataURL('image/png');
                window.parent.postMessage({
                    type: 'canvas-data',
                    data: imageData
                }, '*');
            }
            
            // Add event listener to send data when predict button is clicked
            document.addEventListener('DOMContentLoaded', function() {
                const predictButton = parent.document.querySelector('button:contains("Predict Digit")');
                if (predictButton) {
                    predictButton.addEventListener('click', sendCanvasData);
                }
            });
        </script>
        """,
        height=0,
    )
    
    # Handle canvas data received from JavaScript
    if st.session_state.prediction_made:
        # Show the user input for true label
        user_label = st.number_input("What was the actual digit? (Optional)", min_value=0, max_value=9, step=1)
        
        if st.button("Submit Feedback"):
            # Update the database with user label
            if log_prediction(st.session_state.prediction, st.session_state.confidence, user_label):
                st.success("Feedback submitted successfully!")
            else:
                st.error("Failed to submit feedback")
            
            # Reset the prediction state
            st.session_state.prediction_made = False
            st.experimental_rerun()

if __name__ == "__main__":
    try:
        # Add Streamlit Components for JavaScript communication
        from streamlit import components
        
        # Run the app
        main()
    except Exception as e:
        st.error(f"Application error: {e}")