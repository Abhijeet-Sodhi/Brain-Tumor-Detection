# Brain-Tumor-Detection 👨🏻‍⚕️🧠 
training a convolutional neural network (CNN) to classify brain MRI images as having a tumor or not, using a labeled dataset of 3000 images. It then deploys a Flask web app to upload MRI images, predict the presence of a tumor, and display results.

## Credits 🤖
[![PART 03: Brain Tumor Detection Using Deep Learning | Python Tensorflow Keras | KNOWLEDGE DOCTOR](https://www.youtube.com/watch?v=nmp33q0xXHM&list=LL.jpg)](https://www.youtube.com/watch?v=nmp33q0xXHM&list=LL) - 
**KNOWLEDGE DOCTOR**.
The base code for this project was adapted from KNOWLEDGE DOCTOR. While the original concept and code were used as a foundation, several modifications were made to suit the specific functionality and features of this Brain Tumor detection model.

## Demo 🎬

**Training:**

**Inference:**

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

![image](https://github.com/user-attachments/assets/7d95a7a5-946b-4d10-be43-2ffe93cd4a6c)

#### Input Layer: Reading the MRI Image
- The input to the CNN is an MRI image resized to a fixed dimension (e.g., 64x64x3).
- Each pixel's intensity value is normalized to a range of **[0, 1]** to standardize the input and facilitate faster learning.

#### Feature Extraction: Convolutional Layers
- **Convolution Operation:** Small filters (kernels) slide over the image to detect basic patterns like edges, corners, or textures.
- **Activation Function (ReLU):** ReLU adds non-linearity, allowing the model to learn complex patterns beyond simple ones.
- **Pooling (MaxPooling):** Reduces the image size by keeping only the most important features, making the model faster and more efficient.
- **Example:** The first convolutional layer detects simple edges, and deeper layers identify more complex patterns like shapes or regions related to tumors.

#### Flattening: Converting Features to a Single Vector
- The output of the convolutional and pooling layers is a **multi-dimensional feature map**.
- This map is flattened into a **one-dimensional vector** to connect with the dense (fully connected) layers for classification.

#### Classification: Fully Connected Layers
- The dense layers act as the decision-making part of the network.
- They take the extracted features and learn to associate them with specific labels:
  **Class 0:** No Tumor Detected.
  **Class 1:** Tumor Detected.
- The final output is a vector of probabilities (e.g., [0.8, 0.2]), where each value indicates the likelihood of belonging to a specific class.

#### Softmax Layer: Output Interpretation
- The **softmax activation function** ensures that the output probabilities for the two classes sum to 1.
- The model predicts the class with the highest probability.

#### Inference: Using the Model
- During prediction, the trained model processes a new MRI image, extracts features, and classifies the image based on patterns it learned during training.
- If the detected patterns match those of tumor-affected images, the model predicts **"Tumor Detected."** Otherwise, it predicts **"No Tumor Detected."**


