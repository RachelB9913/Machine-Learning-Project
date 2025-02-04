# Machine Learning Analysis of LeetCode Problems

## Overview

This project applies machine learning techniques to analyze and predict various attributes of LeetCode problems. The dataset used is the [Leetcode Problem Dataset](https://www.kaggle.com/datasets/gzipchrist/leetcode-problem-dataset), containing 1,825 problem entries with 17 features.

## Objectives

- Explore the dataset to uncover relationships between different features.
- Build predictive models for:
  - Problem difficulty classification
  - Related topics prediction
  - Number of accepted submissions prediction
  - Likes and dislikes prediction
- Develop an interactive UI for user-friendly access to these predictions.

## Data Description

The dataset consists of the following key features:

- **Id**: Problem ID
- **Title & Description**: Name and textual description of the problem
- **Is\_Premium**: Whether a premium subscription is required (1 = Yes, 0 = No)
- **Difficulty**: Easy, Medium, or Hard
- **Acceptance\_rate**: Frequency of correct submissions
- **Companies & Related Topics**: Companies that ask the problem and relevant topics
- **Likes & Dislikes**: Community engagement metrics
- **Similar Questions**: Related problems and difficulty levels

## Data Processing

Several preprocessing steps were applied:

- Multi-hot encoding for categorical variables (e.g., `related_topics`, `companies`)
- Label encoding for `difficulty` and `title`
- Conversion of numerical shorthand (e.g., `K` for thousands, `M` for millions)
- Text preprocessing (lemmatization, stopword removal, TF-IDF vectorization)

## Machine Learning Models

Different ML models were used for various tasks:

### 1. Difficulty Classification

- **Models Used:** Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, XGBoost, and a simple Neural Network.
- **Best Performing Model:** Random Forest.
- **Feature Importance:** `is_premium`, `acceptance_rate`, `rating`, `discuss_count`, and top words from TF-IDF.

### 2. Related Topics Prediction (Multi-label Classification)

- **Models Used:** Logistic Regression (One-vs-Rest), SVM, Random Forest, KNN, Gradient Boosting, XGBoost.
- **Threshold Optimization:** Used Stratified K-Fold Cross-Validation to optimize prediction thresholds per label.
- **Best Performing Model:** Random Forest.

### 3. Predicting Accepted Submissions

- **Model Used:** Linear Regression.
- **Key Features:** `submissions`, `difficulty`, `discuss_count`, `is_premium`.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE).

### 4. Predicting Likes & Dislikes

- **Models Used:** Linear Regression, Random Forest, Gradient Boosting, XGBoost.
- **Best Performing Model:** Random Forest for likes and XGBoost for dislikes.
- **Evaluation Metrics:** RMSE, MAE.

## User Interface

An interactive UI was developed using [Gradio](https://huggingface.co/spaces/noa151/LeetCodePredictions). Users can input problem attributes and receive real-time predictions. Features include:

- Predicting related topics
- Predicting problem difficulty
- Predicting acceptance rate
- Predicting likes and dislikes

## Future Work

- Enhance neural network models with deep learning architectures
- Implement a recommendation system for personalized problem suggestions
- Improve NLP-based embeddings using transformer models (e.g., BERT)

## Links

- **Dataset:** [Kaggle - Leetcode Problem Dataset](https://www.kaggle.com/datasets/gzipchrist/leetcode-problem-dataset)
- **User Interface:** [Gradio UI](https://huggingface.co/spaces/noa151/LeetCodePredictions)

## Contributors

- Noa Fishman
- Rachel Belokopytov

