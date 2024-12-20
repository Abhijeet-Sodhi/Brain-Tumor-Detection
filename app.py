import os
import warnings
import logging
from flask import Flask, request, render_template, send_from_directory
from keras.models import load_model
from PIL import Image
import numpy as np
import uuid

# Suppress unnecessary TensorFlow and warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # Suppress TensorFlow info logs
logging.getLogger('absl').setLevel(logging.ERROR)   # Suppress absl warning messages
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# Flask application setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'                           # file where uplods are stored temporarily
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.logger.setLevel(logging.WARNING)                # Suppress Flask development server logs

# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    img = Image.open(image_path)
    img = img.resize((64, 64))                      # Resize the image to match the model's input size
    img = np.array(img) / 255.0                     # Converts image to a NumPy array and normalizes pixel values to [0, 1].
    img = np.expand_dims(img, axis=0)               # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None                               # Stores the prediction result.
    uploaded_file_path = None                       # Stores the path of the uploaded image.
    status_class = None                             # Class to define the text color.

    if request.method == 'POST':                    # Executes if the request is a POST (file upload).
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded') # Renders an error message.

        file = request.files['file']                # Retrieves the uploaded file.

        if file.filename == '':                     # Checks if filename is empty.
            return render_template('index.html', message='No file selected')

        if file:                                    # If a valid file is uploaded.
            # Generate a unique filename
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file to the uploads folder
            file.save(file_path)
            uploaded_file_path = unique_filename    # Store the filename, not the full path
            
            # Preprocess the image and make prediction
            input_img = preprocess_image(file_path)
            predict_x = model.predict(input_img)    # Predict the probabilities for each class.
            result = np.argmax(predict_x, axis=1)   # Gets the index of the class with the highest probability.

            # Map the prediction result to a human-readable label
            if result[0] == 1:
                prediction = "Tumor Detected"
                status_class = "tumor"             # Assign red text class
            else:
                prediction = "No Tumor Detected"
                status_class = "no-tumor"          # Assign green text class
    
    # Render the home page with the prediction result, status class, and uploaded image path.
    return render_template('index.html', prediction=prediction, image_path=uploaded_file_path, status_class=status_class)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
