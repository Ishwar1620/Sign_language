# Indian Sign Language Recognition

This project focuses on recognizing Indian Sign Language (ISL) gestures using Convolutional Neural Networks (CNNs) and transfer learning with the VGG16 model. The model is trained to recognize gestures representing the A-Z alphabet in ISL.

## Overview

- **Model Architecture:** Transfer learning is employed using the VGG16 architecture as the base model, with the final layers customized for the specific sign language recognition task.

- **Alphabets:** The model is trained to recognize gestures corresponding to the A-Z alphabet in Indian Sign Language.

- **Framework:** TensorFlow with Keras is used for model development and training.

- **Real-Time Recognition:** The trained model supports real-time recognition using live camera feed.

## Setup

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Ishwar1620/Sign_language.git
   cd Sign_language
   ```


3. **Download Pre-trained VGG16 Weights:**
   Download the pre-trained VGG16 weights from [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) and place them in the project directory.

4. **Train the Model:**
   ```bash
   python train_model.py
   ```

5. **Run Real-Time Recognition:**
   ```bash
   python main.py
   ```

## Files and Directories

- **`train.py`:** Script for training the CNN model using the VGG16 backbone.

- **`main.py`:** Script for real-time recognition using the trained model and live camera feed.


## Usage

1. **Training:**
   - Run `train.py` to train the model on the provided dataset. Adjust hyperparameters as needed.

2. **Real-Time Recognition:**
   - After training, use `main.py` to run real-time recognition with a live camera feed.

3. **Adjustments:**
   - Modify the code as needed for different datasets, gestures, or model architectures.

## Acknowledgments

- The model architecture is based on the VGG16 model by the Visual Geometry Group at Oxford. (Original source: https://github.com/fchollet/deep-learning-models)

- Dataset credits: https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset

Pretrained Model file : https://drive.google.com/file/d/1d8PPnJc_sT4kytMeQhLNblRuSFCPIa0c/view
