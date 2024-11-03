# HusseinYaacoub7 - Iris Flower Neural Network

## Project Overview
This project implements a neural network model to classify Iris flowers into one of three species: **Iris setosa**, **Iris virginica**, and **Iris versicolor**. The model is trained on the well-known Iris dataset, a benchmark dataset in pattern recognition and machine learning literature. The dataset contains 150 samples, with four features for each sample: **sepal length**, **sepal width**, **petal length**, and **petal width**. 

## Dataset
- **Source:** The Iris dataset is available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) and other open-source data platforms.
- **Features:**
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **Target:** Species of the Iris flower (Setosa, Virginica, Versicolor)

## Project Structure
- **Data Preprocessing:** The dataset is loaded and preprocessed using standardization for optimal neural network performance.
- **Model Architecture:** The model is a feed-forward neural network built with Keras, utilizing dense layers with dropout and batch normalization to improve generalization and reduce overfitting.
- **Training and Evaluation:** The model is trained using RMSprop with callbacks for early stopping and learning rate reduction to enhance performance. The accuracy is then evaluated on a test dataset.

## Requirements
To run the code, you'll need to install the following Python packages:
- `tensorflow`
- `pandas`
- `scikit-learn`
- `numpy`

You can install the necessary packages using:
```bash
pip install tensorflow pandas scikit-learn numpy

This markdown can be directly copied into your `README.md` file on GitHub for clear, formatted documentation of your project. Let me know if you'd like any further edits!
