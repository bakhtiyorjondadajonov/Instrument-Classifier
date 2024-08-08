# Instrument-Classifier
## Overview

The **Instrument Detector** is a web application built with Streamlit and Fastai for image classification. This application distinguishes between three categories of objects:
- **Medical Instruments** 🩺🩻
- **Musical Instruments** 🥁🎸
- **Kitchen Utensils** 🥣🍽️

Leveraging deep learning techniques and the fastai library, it uses the comprehensive Open Images Dataset to achieve high accuracy and robustness. This project demonstrates the power of transfer learning and modern neural network architectures in practical image classification tasks.

## Features

- **Image Classification:** Upload an image to classify it into one of the three categories.
- **Visual Output:** Displays the uploaded image, predicted category, and the probability of the prediction.
- **Interactive Visualization:** Shows a bar chart of classification probabilities for better insight.

## How to Use

1. **Visit the Deployed Application:**
   Access the deployed application at [Instrument Detector](https://instrument-classifier-bakhtiv1.streamlit.app/).

2. **Upload an Image:**
   Use the file uploader to upload an image in one of the following formats: `.jpg`, `.jpeg`, `.png`, `.gif`, or `.svg`.

3. **View Results:**
   The application will display:
   - The uploaded image
   - The predicted category
   - The probability of the prediction
   - A bar chart visualizing the probabilities for all possible classifications
