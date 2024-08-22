# Email Spam Classification

This project involves building a model to classify emails as either "Spam" or "Ham" (not spam) using various machine learning algorithms. The main steps include data preprocessing, vectorization, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
  - [Naive Bayes](#naive-bayes)
  - [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
  - [Neural Network](#neural-network)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

In this project, we will:

1. Load and preprocess the email dataset.
2. Transform the text data using TF-IDF vectorization.
3. Train and evaluate different machine learning models including Naive Bayes, SVM, and a Neural Network.
4. Visualize the results and performance metrics.

## Dataset

The dataset consists of labeled email data, with labels indicating whether the email is spam (`1`) or ham (`0`). It is loaded from a CSV file and consists of 83,448 entries.

## Preprocessing

- The labels are converted to binary values: `1` for spam and `0` for ham.
- The text data is cleaned by removing unnecessary characters and normalizing spaces.
- The dataset is split into training and testing sets.

## Model Training

### Naive Bayes

- **Vectorization**: Text data is vectorized using TF-IDF.
- **Training**: A Multinomial Naive Bayes model is trained on the training dataset.
- **Evaluation**: The model is evaluated using metrics like accuracy, precision, recall, and F1 score. Confusion matrices are also plotted.

### Support Vector Classifier (SVC)

- **Parameter Tuning**: GridSearchCV is used to find the best parameters (`gamma` and `C`) for the SVC model.
- **Training and Evaluation**: The best SVC model is trained and evaluated similarly to the Naive Bayes model.

### Neural Network

- **Architecture**: A simple neural network with two hidden layers is implemented using TensorFlow and Keras.
- **Training**: The model is trained using the Adam optimizer with a binary cross-entropy loss function.
- **Evaluation**: The model's performance is evaluated on the test data.

## Results

The models are compared based on their performance metrics. Plots and visualizations help in understanding the effectiveness of each model.

## Conclusion

- **Naive Bayes**: Provides a good balance between simplicity and accuracy.
- **SVC**: Shows a slight improvement when fine-tuned with GridSearchCV.
- **Neural Network**: Offers potential but requires careful tuning and more data to outperform traditional models.

By tuning hyperparameters and experimenting with different models, this project demonstrates how machine learning can effectively classify emails as spam or ham.
