from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, jaccard_score, hamming_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import joblib
import random
import os
import torch
import tensorflow as tf
from data_preparation import data_preparation


def set_all_seeds(seed=42):
    """Set all seeds to make results reproducible"""
    random.seed(seed)  # Python
    np.random.seed(seed)  # Numpy
    random.seed(seed)  # Sklearn
    tf.random.set_seed(seed)  # Tensorflow
    torch.manual_seed(seed)  # Torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Environment


class MultiLabelThresholdOptimizer:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.optimal_thresholds = {}

    def find_optimal_thresholds(self, y_true, y_pred_proba):
        """Find optimal threshold for each label using F1 score"""
        n_labels = y_true.shape[1]
        thresholds = np.zeros(n_labels)

        for label in range(n_labels):
            best_f1 = 0
            best_threshold = 0.5

            # Use fixed thresholds to ensure reproducibility
            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred = (y_pred_proba[:, label] >= threshold).astype(int)
                f1 = f1_score(y_true[:, label], y_pred, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            thresholds[label] = best_threshold

        return thresholds

    def fit(self, X, y, model, model_name):
        """Find and save optimal thresholds using cross validation"""
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_thresholds = []

        for train_idx, val_idx in kf.split(X, y[:, 0]):
            X_val = X[val_idx]
            y_val = y[val_idx]

            if isinstance(X, np.ndarray):
                y_pred_proba = model.predict_proba(X_val)
            else:
                y_pred_proba = model.predict_proba(X_val)

            fold_thresholds.append(self.find_optimal_thresholds(y_val, y_pred_proba))

        final_thresholds = np.median(fold_thresholds, axis=0)
        self.optimal_thresholds[model_name] = final_thresholds

        return final_thresholds

    def predict(self, model, X, model_name):
        if model_name not in self.optimal_thresholds:
            raise ValueError(f"No thresholds found for model: {model_name}")

        if isinstance(X, np.ndarray):
            y_pred_proba = model.predict_proba(X)
        else:
            y_pred_proba = model.predict_proba(X)

        thresholds = self.optimal_thresholds[model_name]
        y_pred = np.zeros_like(y_pred_proba)

        for label in range(y_pred_proba.shape[1]):
            y_pred[:, label] = (y_pred_proba[:, label] >= thresholds[label]).astype(int)

        return y_pred


def compare_models(results):
    """ Compare models across all metrics and provide rankings.
    Now includes rankings for:
    - Precision
    - Recall
    - F1 Score
    - Subset Accuracy
    - Hamming Accuracy
    - Jaccard Score """
    metrics = ['precision', 'recall', 'f1', 'subset_accuracy', 'hamming_accuracy', 'jaccard_score']
    rankings = {metric: {} for metric in metrics}

    # Rank models for each metric
    for metric in metrics:
        sorted_models = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            rankings[metric][model_name] = rank

    # Compute average ranking across all metrics
    average_rankings = {}
    for model_name in results.keys():
        model_ranks = [rankings[metric][model_name] for metric in metrics]
        average_rankings[model_name] = sum(model_ranks) / len(metrics)

    # Sort models by average ranking (lower is better)
    final_ranking = sorted(average_rankings.items(), key=lambda x: x[1])

    # Print detailed comparison
    print("\n🏆 Model Comparison Results:")
    print("\n📊 Detailed Metrics and Rankings:")
    headers = ['Model', 'Precision', 'Recall', 'F1 Score', 'Subset Acc', 'Hamming Acc', 'Jaccard', 'Avg Rank']
    print('-' * 120)
    print(f"{headers[0]:<24} {headers[1]:<12} {headers[2]:<11} {headers[3]:<10} {headers[4]:<10} {headers[5]:<12} {headers[6]:<10} {headers[7]:<8}")
    print('-' * 120)

    for model_name in results.keys():
        metrics = results[model_name]
        print(f"{model_name:<20} "
              f"{metrics['precision']:>11.3f} "
              f"{metrics['recall']:>11.3f} "
              f"{metrics['f1']:>11.3f} "
              f"{metrics['subset_accuracy']:>11.3f} "
              f"{metrics['hamming_accuracy']:>11.3f} "
              f"{metrics['jaccard_score']:>11.3f} "
              f"{average_rankings[model_name]:>8.2f}")

    print('-' * 120)

    # Print final rankings
    print("\n🎯 Final Model Rankings (based on average performance across all metrics):")
    for rank, (model_name, avg_rank) in enumerate(final_ranking, 1):
        print(f"{rank}. {model_name:<20} (Average Rank: {avg_rank:.2f})")

    # Identify best model
    best_model = final_ranking[0][0]
    print(f"\n🥇 Best Overall Model: {best_model}")
    print("\n📌 Detailed strengths of the best model:")
    print(f"   - Precision: {results[best_model]['precision']:.3f}")
    print(f"   - Recall: {results[best_model]['recall']:.3f}")
    print(f"   - F1 Score: {results[best_model]['f1']:.3f}")
    print(f"   - Subset Accuracy: {results[best_model]['subset_accuracy']:.3f}")
    print(f"   - Hamming Accuracy: {results[best_model]['hamming_accuracy']:.3f}")
    print(f"   - Jaccard Score: {results[best_model]['jaccard_score']:.3f}")

    return best_model, results[best_model]


def save_best_model_info(best_model_name, model_metrics, threshold):
    """ Save information about the best model """
    best_model_info = {
        'model_name': best_model_name,
        'metrics': model_metrics,
        'threshold': threshold
    }
    joblib.dump(best_model_info, 'best_model_related_topics_info.pkl')


def evaluate_model_related(y_test, y_pred, model_name):
    """Evaluate model performance with additional accuracy metrics"""
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Subset accuracy (Exact match ratio)
    subset_accuracy = accuracy_score(y_test, y_pred)

    # Hamming accuracy (1 - Hamming loss)
    hamming_acc = 1 - hamming_loss(y_test, y_pred)

    # Jaccard similarity score (macro averaged across all labels)
    jaccard_macro = jaccard_score(y_test, y_pred, average='samples', zero_division=0)

    return {
        'precision': precision_weighted,
        'recall': recall_weighted,
        'f1': f1_weighted,
        'subset_accuracy': subset_accuracy,
        'hamming_accuracy': hamming_acc,
        'jaccard_score': jaccard_macro
    }


def related_topics_prediction():
    # Set all seeds for reproducibility
    SEED = 42
    set_all_seeds(SEED)

    warnings.filterwarnings("ignore", category=UserWarning)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("data.csv")
    df = df.dropna(subset=['related_topics'])
    df['description'] = df['description'].str.lower().fillna('')
    df['related_topics'] = df['related_topics'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Extract unique topics
    all_possible_topics = sorted(set(topic for topics in df['related_topics'] for topic in topics))
    print(f"\n✅ Found {len(all_possible_topics)} unique topics.")

    # Prepare features and labels with deterministic behavior
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        stop_words='english'
    )
    X = vectorizer.fit_transform(df['description'])
    joblib.dump(vectorizer, 'related_topics_vectorizer.pkl')

    mlb = MultiLabelBinarizer(classes=all_possible_topics)
    y = mlb.fit_transform(df['related_topics'])
    joblib.dump(mlb, 'related_topics_label_binarizer.pkl')

    # Split dataset with fixed random state
    X_train, X_test, y_train, y_test, desc_train, desc_test = train_test_split(
        X, y, df['description'], test_size=0.2, random_state=SEED, shuffle=True
    )

    # Initialize models with fixed random states
    models = {
        'SVM': OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=SEED)),
        'Logistic_Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=SEED)),
        'Random_Forest': OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=SEED)),
        'KNN': OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5)),
        'Gradient_Boosting': OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=SEED)),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=SEED,
            seed=SEED
        )
    }

    # Initialize threshold optimizer
    optimizer = MultiLabelThresholdOptimizer(random_state=SEED)
    results = {}
    results_threshold = {}

    # Train and optimize each model
    for model_name, model in models.items():
        print(f"\n⏳ Training {model_name} model...")
        model.fit(X_train, y_train)

        print(f"Finding optimal thresholds for {model_name}...")
        thresholds = optimizer.fit(X_train.toarray() if not isinstance(X_train, np.ndarray) else X_train,
                                   y_train, model, model_name)

        results_threshold[model_name] = thresholds
        y_pred = optimizer.predict(model, X_test, model_name)
        results[model_name] = evaluate_model_related(y_test, y_pred, model_name)

    print("\nSelecting best model...")
    best_model_name, best_model_metrics = compare_models(results)
    save_best_model_info(best_model_name, best_model_metrics, results_threshold[best_model_name])
    trained_best_model = models[best_model_name]

    # If it's a GridSearchCV model, extract the best estimator
    if isinstance(trained_best_model, GridSearchCV):
        trained_best_model = trained_best_model.best_estimator_

    joblib.dump(trained_best_model, "best_related_topics_model.pkl")
    print(f"✅ Best trained model saved as best_related_topics_model.pkl")

    # Display sample predictions with fixed indices
    print("\n📌 Sample Predictions with Optimized Thresholds:")
    num_samples = 5
    # Use fixed indices instead of random sampling
    sample_indices = list(range(min(5, len(X_test.toarray()))))

    for idx in sample_indices:
        print(f"\nDescription: {desc_test.iloc[idx][:100]}...")
        print(f"✅ True Topics: {', '.join(mlb.inverse_transform(np.array([y_test[idx]]))[0])}")

        for model_name in models.keys():
            y_pred = optimizer.predict(models[model_name], X_test[idx], model_name)
            predicted_labels = mlb.inverse_transform(y_pred)[0]
            print(f"🔮 Predicted ({model_name}): {', '.join(predicted_labels) if predicted_labels else 'None'}")

    print("\n✅ Training and evaluation completed. Models and thresholds saved.")


def train_and_evaluate(the_data, top_in_tfidf, target_column, metrics_dict, model_type="linear_regression"):
    # Select features
    base_features = ["difficulty", "is_premium", "frequency", "discuss_count", "accepted", "submissions"]
    if target_column == "likes":
        base_features.append("dislikes")
    else:
        base_features.append("likes")

    X = the_data[base_features]
    X = pd.concat([X, the_data[top_in_tfidf]], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, the_data[[target_column]], test_size=0.2, random_state=42)

    y_train_target = y_train[target_column]
    y_test_target = y_test[target_column]

    # Apply Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Invalid model_type. Choose 'linear' or 'random_forest'.")

    # Train model
    model.fit(X_train_scaled, y_train_target)

    # Predict
    y_pred = model.predict(X_test_scaled)

    if target_column == "likes" and model_type == "random_forest":
        feature_names = list(X_train.columns)
        joblib.dump((model, feature_names), "likes_random_forest_regression_model.pkl")
        print(f"✅ likes_random_forest_regression_model.pkl was saved successfully.\n")


    # Evaluate model
    mae = mean_absolute_error(y_test_target, y_pred)
    mse = mean_squared_error(y_test_target, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_target, y_pred)

    print(f"{target_column} - {model_type.replace('_', ' ').title()} Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test_target, y=y_pred, alpha=0.7, color='#A6CDC6', label='Predicted vs Actual')  # Dot color
    plt.plot([y_test_target.min(), y_test_target.max()], [y_test_target.min(), y_test_target.max()],
             color='#DDA853', lw=2, label='Ideal Fit')  # Line color
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{target_column} Prediction Performance - {model_type.replace('_', ' ').title()}")
    plt.legend()
    plt.show()

    metrics_dict[model_type] = {"MAE": mae, "MSE": mse, "RMSE": rmse}
    return metrics_dict


# for the boosting algorithms
def evaluate_model(y_test, y_pred, model_name, target_column, metrics_dict):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    plt.figure(figsize=(8, 6))

    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#A6CDC6', label='Predicted vs Actual')  # Dot color
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#DDA853', lw=2,
             label='Ideal Fit')  # Line color
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{target_column} Prediction Performance - {model_name}")
    plt.legend()
    plt.show()

    metrics_dict[model_name] = {"MAE": mae, "MSE": mse, "RMSE": rmse}
    return metrics_dict


def plot_comparison_histogram(metrics_dict, target_column):
    metrics = ["RMSE", "MAE"]
    models = list(metrics_dict.keys())
    colors = ["#A6CDC6", "#DDA853"]  # Blue for RMSE, Orange for MAE

    # Prepare data for plotting
    values = {metric: [metrics_dict[model][metric] for model in models] for metric in metrics}

    x = np.arange(len(models))  # X locations for models
    width = 0.3  # Wider bars for better visibility

    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = [-width / 2, width / 2]  # Positions for the two bars

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offsets[i], values[metric], width, label=metric, color=colors[i])

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.0f}",
                    ha='center', va='bottom', fontsize=10, rotation=0)

    ax.set_xlabel("Models")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Comparison of Models for {target_column.capitalize()}")

    # Ensure x-axis labels are centered with bars
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center')

    ax.legend()
    plt.show()


def get_best_model(metrics_dict, target_column):
    # Find the model with the minimum MAE and RMSE
    best_mae_model = min(metrics_dict, key=lambda model: metrics_dict[model]["MAE"])
    best_rmse_model = min(metrics_dict, key=lambda model: metrics_dict[model]["RMSE"])
    if best_rmse_model == best_mae_model:
        print(f"\n🔥 Best model for {target_column} is {best_rmse_model}")


# Wrapper functions
def linear_regression_for_likes(the_data, top_in_tfidf, metrics_dict):
    return train_and_evaluate(the_data, top_in_tfidf, "likes", metrics_dict, model_type="linear_regression")


def linear_regression_for_dislikes(the_data, top_in_tfidf, metrics_dict):
    return train_and_evaluate(the_data, top_in_tfidf, "dislikes", metrics_dict, model_type="linear_regression")


def random_forest_for_likes(the_data, top_in_tfidf, metrics_dict):
    train_and_evaluate(the_data, top_in_tfidf, "likes", metrics_dict, model_type="random_forest")


def random_forest_for_dislikes(the_data, top_in_tfidf, metrics_dict):
    train_and_evaluate(the_data, top_in_tfidf, "dislikes", metrics_dict, model_type="random_forest")


def gradient_boosting_for_likes(the_data, top_in_tfidf, metrics_dict):
    X = the_data[["difficulty", "acceptance_rate", "frequency", "discuss_count", "accepted", "submissions", "dislikes"]]
    X = pd.concat([X, the_data[top_in_tfidf]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, the_data[["likes"]], test_size=0.2, random_state=42)

    y_train_likes = y_train["likes"]
    y_test_likes = y_test["likes"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train_likes)
    y_pred = model.predict(X_test_scaled)

    return evaluate_model(y_test_likes, y_pred, "Gradient Boosting", "likes", metrics_dict)


def gradient_boosting_for_dislikes(the_data, top_in_tfidf, metrics_dict):
    X = the_data[["difficulty", "acceptance_rate", "frequency", "discuss_count", "accepted", "submissions", "likes"]]
    X = pd.concat([X, the_data[top_in_tfidf]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, the_data[["dislikes"]], test_size=0.2, random_state=42)

    y_train_dislikes = y_train["dislikes"]
    y_test_dislikes = y_test["dislikes"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train_dislikes)
    y_pred = model.predict(X_test_scaled)

    return evaluate_model(y_test_dislikes, y_pred, "Gradient Boosting", "dislikes", metrics_dict)


def xgboost_for_likes(the_data, top_in_tfidf, metrics_dict):
    X = the_data[["difficulty", "acceptance_rate", "frequency", "discuss_count", "accepted", "submissions", "dislikes"]]
    X = pd.concat([X, the_data[top_in_tfidf]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, the_data[["likes"]], test_size=0.2, random_state=42)

    y_train_likes = y_train["likes"]
    y_test_likes = y_test["likes"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train_likes)
    y_pred = model.predict(X_test_scaled)

    return evaluate_model(y_test_likes, y_pred, "XGBoost", "likes", metrics_dict)


def xgboost_for_dislikes(the_data, top_in_tfidf, metrics_dict):
    X = the_data[["difficulty", "acceptance_rate", "frequency", "discuss_count", "accepted", "submissions", "likes"]]
    X = pd.concat([X, the_data[top_in_tfidf]], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, the_data[["dislikes"]], test_size=0.2, random_state=42)

    y_train_dislikes = y_train["dislikes"]
    y_test_dislikes = y_test["dislikes"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train_dislikes)
    y_pred = model.predict(X_test_scaled)

    feature_names = list(X_train.columns)
    joblib.dump((model, feature_names), "dislikes_XGB_regression_model.pkl")
    print(f"✅ dislikes_XGB_regression_model.pkl was saved successfully.\n")

    return evaluate_model(y_test_dislikes, y_pred, "XGBoost", "dislikes", metrics_dict)


# linear regression that predict the amount of accepted answers from submissions, difficulty, discuss_count
def accepted_submissions_regression(df):
    # Select features (X) and target (y)
    X = df[["submissions", "difficulty", "discuss_count", "is_premium"]]  # Use the numeric difficulty column
    y = df["accepted"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Assuming `X_train` is a DataFrame used to train the model
    feature_names = list(X_train.columns)

    # Save both the model and feature names
    joblib.dump((model, feature_names), "accepted_submissions_regression_model.pkl")
    print(f"✅ accepted_submissions_regression_model.pkl was saved successfully.\n")

    # Model evaluation
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("R-squared Score (R²):", r2_score(y_test, y_pred))

    # Function to predict accepted submissions
    def predict_accepted(submitted, difficulty):
        input_data = [[submitted, difficulty]]  # Use numeric encoding for difficulty
        predicted_value = model.predict(input_data)[0]
        return max(0, round(predicted_value))  # Ensure predictions are non-negative

    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#DDA853', linewidth=2, label="Regression Line")
    plt.scatter(y_test, y_pred, color="#A6CDC6", label="Predicted vs Actual")
    plt.xlabel("Actual Accepted Submissions")
    plt.ylabel("Predicted Accepted Submissions")
    plt.title("Actual vs Predicted Accepted Submissions")
    plt.legend()
    plt.tight_layout()
    plt.show()


class SklearnXGBClassifier(XGBClassifier, BaseEstimator, ClassifierMixin):
    def __sklearn_tags__(self):
        tags = super()._get_tags()
        tags.update({"non_deterministic": True})
        return tags


# logistics regression to classify the questions by levels (hard, medium, easy)
def difficulty_classification(df, top_in_tfidf, related_topics, nn_result):
    # Feature Engineering
    X = df[["is_premium", "acceptance_rate", "rating", "discuss_count"]]
    X = pd.concat([X, df[top_in_tfidf]], axis=1)
    X = pd.concat([X, df[related_topics]], axis=1)
    y = df["difficulty"]

    # Handling Class Imbalance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Classifiers with Hyperparameter Tuning
    param_grid_rf = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20]
    }
    rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    rf_grid = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring="accuracy")

    classifiers = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression()),
        "Random Forest": rf_grid,
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
        "XGBoost": SklearnXGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=10, gamma=0.1, colsample_bytree=0.7, subsample=0.8)
    }

    results = {}
    results[nn_result["model_name"]] = nn_result

    for name, model in classifiers.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "predictions": y_pred
        }

        print(f"{name} - Accuracy: {accuracy:.2%}")
        print(classification_report(y_test, y_pred))

    # Find Best Model
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model_accuracy = results[best_model_name]["accuracy"]
    best_y_pred = results[best_model_name]["predictions"]
    print(f"\n🏆 Best Model: {best_model_name} - Accuracy: {best_model_accuracy:.2%}")

    # Save the best model
    trained_best_model = classifiers[best_model_name]
    # If it's a GridSearchCV model, extract the best estimator
    if isinstance(trained_best_model, GridSearchCV):
        trained_best_model = trained_best_model.best_estimator_

    joblib.dump(trained_best_model, "level_classifier_model.pkl")
    print(f"✅ Best trained model saved as level_classifier_model.pkl")

    # Plot Accuracy Comparison
    plt.figure(figsize=(10, 5))
    bars = plt.bar(results.keys(), [res["accuracy"] for res in results.values()], color="#A6CDC6")
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy Comparison")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2%}", ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 🎯 Visualization for the best model
    results_df = pd.DataFrame({
        'Index': range(len(y_test)),
        'Actual': y_test.values,
        'Predicted': best_y_pred
    })

    difficulty_levels = [0, 1, 2]
    level_names = ['Easy', 'Hard', 'Medium']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Model Performance by Difficulty Level ({best_model_name})', fontsize=16, y=1.05)

    for idx, (level, name) in enumerate(zip(difficulty_levels, level_names)):
        mask_actual = results_df['Actual'] == level
        level_accuracy = accuracy_score(
            results_df[mask_actual]['Actual'],
            results_df[mask_actual]['Predicted']
        )

        axes[idx].scatter(
            results_df[mask_actual]['Index'],
            results_df[mask_actual]['Actual'],
            alpha=0.6,
            s=100,
            marker='o',
            color='#A6CDC6'
        )
        axes[idx].scatter(
            results_df[mask_actual]['Index'],
            results_df[mask_actual]['Predicted'],
            alpha=0.6,
            s=100,
            marker='x',
            color='#3B6790'
        )

        axes[idx].set_title(f'{name} (Level {level})\nAccuracy: {level_accuracy:.2%}')
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('Difficulty Level')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].set_yticks(difficulty_levels)

    plt.tight_layout()
    plt.show()

    return results


class DifficultyDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DifficultyClassifier(nn.Module):
    def __init__(self, input_size):
        super(DifficultyClassifier, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Linear(64, 3)

        # Initialize weights using He initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), total_loss / len(test_loader)


def difficulty_classification_nn(df, top_in_tfidf, related_topics, seed=42):
    set_seed(seed)  # Set the seed for reproducibility

    # Feature Engineering
    X = df[["is_premium", "acceptance_rate", "rating", "discuss_count"]]
    X = pd.concat([X, df[top_in_tfidf]], axis=1)
    X = pd.concat([X, df[related_topics]], axis=1)
    y = df["difficulty"]

    # Handling Class Imbalance
    smote = SMOTE(random_state=seed)
    X, y = smote.fit_resample(X, y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data with validation set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=seed)

    # Create datasets and dataloaders
    train_dataset = DifficultyDataset(X_train, y_train)
    val_dataset = DifficultyDataset(X_val, y_val)
    test_dataset = DifficultyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DifficultyClassifier(input_size=X.shape[1]).to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=15)
    n_epochs = 100

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\nTraining Neural Network...")
    for epoch in range(n_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate
        val_pred, val_true, val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_acc = accuracy_score(val_true, val_pred)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Final evaluation
    y_pred, y_true, _ = evaluate(model, test_loader, criterion, device)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nFinal Test Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color="#A6CDC6")
    ax1.plot(val_losses, label='Validation Loss', color="#3B6790")
    ax1.set_title("Loss Over Time")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Accuracy plot
    ax2.plot(train_accuracies, label='Train Accuracy', color="#A6CDC6")
    ax2.plot(val_accuracies, label='Validation Accuracy', color="#3B6790")
    ax2.set_title("Accuracy Over Time")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Visualization for predictions
    results_df = pd.DataFrame({
        'Index': range(len(y_true)),
        'Actual': y_true,
        'Predicted': y_pred
    })

    difficulty_levels = [0, 1, 2]
    level_names = ['Easy', 'Hard', 'Medium']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Neural Network Performance by Difficulty Level', fontsize=16, y=1.05)

    for idx, (level, name) in enumerate(zip(difficulty_levels, level_names)):
        mask_actual = results_df['Actual'] == level
        level_accuracy = accuracy_score(
            results_df[mask_actual]['Actual'],
            results_df[mask_actual]['Predicted']
        )

        axes[idx].scatter(
            results_df[mask_actual]['Index'],
            results_df[mask_actual]['Actual'],
            alpha=0.6,
            s=100,
            marker='o',
            color='#A6CDC6'
        )
        axes[idx].scatter(
            results_df[mask_actual]['Index'],
            results_df[mask_actual]['Predicted'],
            alpha=0.6,
            s=100,
            marker='x',
            color='#3B6790'
        )

        axes[idx].set_title(f'{name} (Level {level})\nAccuracy: {level_accuracy:.2%}')
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('Difficulty Level')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].set_yticks(difficulty_levels)

    plt.tight_layout()
    plt.show()

    return {
        "model_name": "Neural Network",
        "accuracy": accuracy,
        "report": classification_report(y_true, y_pred, output_dict=True),
        "predictions": y_pred,
        "model": model,
        "history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accuracies,
            "val_acc": val_accuracies
        }
    }


if __name__ == '__main__':
    file_path = "data.csv"
    data = data_preparation(file_path)

    with open("encoding_metadata.json", "r") as f:
        encoding_metadata = json.load(f)

    metrics_dict_likes = {}
    metrics_dict_dislikes = {}

    # Clean up the columns by removing empty strings
    companies_columns = [col for col in encoding_metadata["companies_columns"] if col != ""]
    related_topics_columns = [col for col in encoding_metadata["related_topics_columns"] if col != ""]
    top_in_tfidf_columns = [col for col in encoding_metadata["top_words"] if col != ""]

    # Load preprocessed data and train the model
    prepared_data_path = "data_with_numerical_encodings.csv"
    prepared_data = pd.read_csv(prepared_data_path)

    print("----------- Linear Regression:")
    linear_regression_for_likes(prepared_data, top_in_tfidf_columns, metrics_dict_likes)
    linear_regression_for_dislikes(prepared_data, top_in_tfidf_columns, metrics_dict_dislikes)
    print("\n-----------  Random Forest:")
    random_forest_for_likes(prepared_data, top_in_tfidf_columns, metrics_dict_likes)
    random_forest_for_dislikes(prepared_data, top_in_tfidf_columns, metrics_dict_dislikes)
    print("\n----------- Gradient Boosting:")
    gradient_boosting_for_likes(prepared_data, top_in_tfidf_columns, metrics_dict_likes)
    gradient_boosting_for_dislikes(prepared_data, top_in_tfidf_columns, metrics_dict_dislikes)
    print("\n----------- XGBoost:")
    xgboost_for_likes(prepared_data, top_in_tfidf_columns, metrics_dict_likes)
    xgboost_for_dislikes(prepared_data, top_in_tfidf_columns, metrics_dict_dislikes)
    # Plot Histograms
    plot_comparison_histogram(metrics_dict_likes, "likes")
    plot_comparison_histogram(metrics_dict_dislikes, "dislikes")
    # Print the best models based on MAE and RMSE
    get_best_model(metrics_dict_likes, "likes")
    get_best_model(metrics_dict_dislikes, "dislikes")

    accepted_submissions_regression(prepared_data)
    results = difficulty_classification_nn(prepared_data, top_in_tfidf_columns, related_topics_columns)
    difficulty_classification(prepared_data, top_in_tfidf_columns, related_topics_columns, results)

    related_topics_prediction()
