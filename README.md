# Brain-Tumor-Detection 👨🏻‍⚕️🧠 
training a convolutional neural network (CNN) to classify brain MRI images as having a tumor or not, using a labeled dataset of 3000 images. It then deploys a Flask web app to upload MRI images, predict the presence of a tumor, and display results.

## Credits 🤖
[![PART 03: Brain Tumor Detection Using Deep Learning | Python Tensorflow Keras | KNOWLEDGE DOCTOR](https://www.youtube.com/watch?v=nmp33q0xXHM&list=LL.jpg)](https://www.youtube.com/watch?v=nmp33q0xXHM&list=LL) - 
**KNOWLEDGE DOCTOR**.
The base code for this project was adapted from KNOWLEDGE DOCTOR. While the original concept and code were used as a foundation, several modifications were made to suit the specific functionality and features of this Brain Tumor detection model.

## Demo 🎬



## The Code files: 📄

**mainTrain.py:** Preprocesses the MRI dataset, builds, trains, and saves the CNN model.

**app.py:** Hosts the Flask web app, handles image uploads, preprocesses them, and serves predictions using the trained model.

**index.html & styles.css:** Define the user interface and styling for the web application.

## Functionality ⚙️
**Training:**
- Trains a CNN with three convolutional layers to classify MRI images into "Tumor Detected" or "No Tumor Detected."
- Saves the trained model as BrainTumor10EpochsCategorical.h5.

**Web Interface:**
- Users upload MRI images via the web app.
- Images are preprocessed, and predictions are made using the trained model.
- The result is displayed along with the uploaded image.

## Installation 💻
To run this project, ensure the following dependencies are installed:

*pip install flask==2.2.2*

*pip install tensorflow==2.18.0*

*pip install keras==3.6.0*

*pip install pillow==10.2.0*

*pip install opencv-python==4.10.0.84*

## Usage 📌
- Clone the repository and navigate to the project folder.
- Place the trained model (BrainTumor10EpochsCategorical.h5) in the root directory.
- Run the Flask application:

  *python app.py*

- Open your browser and navigate to **http://127.0.0.1:5000** to use the web interface.

## Theory: 💡
