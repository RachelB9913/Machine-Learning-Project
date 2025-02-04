import json
import gradio as gr
import joblib
import pandas as pd
from related_topics_prediction import MultiLabelThresholdOptimizer


def convert_to_float(value):
    if 'K' in value:
        return float(value.replace('K', '')) * 1_000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1_000_000
    return float(value)  # If it's already a number


def convert_to_string(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))  # Keep it as an integer if it's below 1,000


def greet(title, description, difficulty, topics, likes, accepted, submission, comments, is_premium, predict):

    x_new = pd.DataFrame([{
        'id': 1,
        'title': str(title),
        'description': str(description),
        'is_premium': 1 if is_premium == "premium" else 0,
        'difficulty': 0 if difficulty == "Easy" else 1 if difficulty == "Hard" else 2,
        'acceptance_rate': convert_to_float(accepted)/convert_to_float(submission),
        'frequency': 0,
        'discuss_count': float(comments),
        'accepted': convert_to_float(accepted),
        'submissions': convert_to_float(submission),
        'companies': [""],
        'related_topics': topics.split(',') if isinstance(topics, str) else topics,
        'likes': convert_to_float(likes),
        'dislikes': 0,
        'rating': convert_to_float(likes) / (convert_to_float(likes) + 0),
        'asked_by_faang': 0,
        'similar_questions': ""
    }])

    # Efficient Multi-Hot Encoding for Companies
    company_data = {company: 1 if company in x_new["companies"].iloc[0] else 0 for company in companies_columns}
    x_new = pd.concat([x_new, pd.DataFrame([company_data])], axis=1)

    x_new = x_new.drop(columns=["companies"])  # Drop original column

    # Efficient Multi-Hot Encoding for Topics
    topic_data = {topic: 1 if topic in x_new["related_topics"].iloc[0] else 0 for topic in the_topics}
    x_new = pd.concat([x_new, pd.DataFrame([topic_data])], axis=1)

    x_new = x_new.drop(columns=["related_topics"])  # Drop original topics column

    # Label encode 'title'
    title_model = joblib.load("title_encoder.pkl")
    x_new['title'] = title_model.fit_transform(x_new['title'])

    if predict == "related topics":
        vectorizer = joblib.load("related_topics_vectorizer.pkl")

        new_tfidf = vectorizer.transform(x_new["description"])

        best_model_info = joblib.load('best_model_related_topics_info.pkl')
        best_model = joblib.load("best_related_topics_model.pkl")
        optimizer = MultiLabelThresholdOptimizer()
        optimizer.optimal_thresholds[best_model_info['model_name']] = best_model_info['threshold']

        predictions = optimizer.predict(best_model, new_tfidf, best_model_info['model_name'])

        mlb = joblib.load("related_topics_label_binarizer.pkl")
        predictions = mlb.inverse_transform(predictions)

        ans = f"the related topics are: {', '.join(map(str, predictions[0]))}"
        return ans

    else:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")

        new_tfidf = vectorizer.transform(x_new["description"])

        # Convert to DataFrame
        new_tfidf_df = pd.DataFrame(new_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        x_new = pd.concat([x_new, new_tfidf_df], axis=1)
        x_new = x_new.drop(columns=['description'])

        if predict == "difficulty level":
            # load the dislike model because there is no dislike in the input
            dislikes_model, feature_names = joblib.load("dislikes_XGB_regression_model.pkl")

            x_new_filtered = x_new[feature_names]  # Select only the required features
            dislike = dislikes_model.predict(x_new_filtered)
            x_new['dislikes'] = dislike[0]
            x_new['rating']: convert_to_float(likes) / (convert_to_float(likes) + dislike[0])

            # Load the model
            class_model = joblib.load("level_classifier_model.pkl")

            # Get feature names from trained model
            trained_feature_names = class_model.named_steps['standardscaler'].get_feature_names_out()

            x_new = x_new[trained_feature_names]  # Reorder and remove extra columns

            # Fill missing columns with 0 (or a suitable default)
            for col in trained_feature_names:
                if col not in x_new:
                    x_new[col] = 0  # or another default value

            x_new = x_new[trained_feature_names]  # Ensure correct order again

            predictions = class_model.predict(x_new)

            if predictions == 1:
                prediction = "Hard"
            elif predictions == 0:
                prediction = "Easy"
            elif predictions == 2:
                prediction = "Medium"

            ans = f"the level difficulty is: {prediction}"
            return ans

        elif predict == "acceptance":
            # Load the model
            accepted_submissions_model, feature_names = joblib.load("accepted_submissions_regression_model.pkl")

            # Assuming `X_new` is a DataFrame with extra features
            x_new_filtered = x_new[feature_names]  # Select only the required features

            predictions = accepted_submissions_model.predict(x_new_filtered)

            ans = f"the accepted is: {convert_to_string(predictions[0])}"
            return ans

        elif predict == "number of likes":
            # Load the model
            likes_model, feature_names = joblib.load("likes_random_forest_regression_model.pkl")

            # Assuming `X_new` is a DataFrame with extra features
            x_new_filtered = x_new[feature_names]  # Select only the required features

            predictions = likes_model.predict(x_new_filtered)

            ans = f"the likes amount is: {convert_to_string(predictions[0])}"
            return ans

        elif predict == "number of dislikes":
            # Load the model
            dislikes_model, feature_names = joblib.load("dislikes_XGB_regression_model.pkl")

            # Assuming `x_new` is a DataFrame with extra features
            x_new_filtered = x_new[feature_names]  # Select only the required features

            predictions = dislikes_model.predict(x_new_filtered)

            ans = f"the dislikes amount is: {convert_to_string(predictions[0])}"
            return ans


with open("encoding_metadata.json", "r") as f:
    encoding_metadata = json.load(f)

the_topics = encoding_metadata["related_topics_columns"]
the_topics.remove("")
companies_columns = encoding_metadata["companies_columns"]
companies_columns.remove("")

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Text(label="Title"), gr.Text(label="Description"),
            gr.Radio(choices=["Easy", "Medium", "Hard"], label="Difficulty Level"),
            gr.Dropdown(the_topics, multiselect=True, label="Related Topics",
                        info="choose all the related topics of this question"),
            gr.Text(label="Likes Amount"),
            gr.Text(label="Accepted Amount"),
            gr.Text(label="Submission Amount"),
            gr.Text(label="Comments Amount"),
            gr.Radio(choices=["premium", "not premium"], label="Is Premium"),
            gr.Radio(choices=["acceptance", "difficulty level", "number of likes", "number of dislikes",
                              "related topics"], label="Please Predict..")
            ],
    outputs=[gr.Text(label="The Prediction")],
    title="LEETCODE PREDICTOR",
    description="please go to the leetcode website (https://leetcode.com/) choose a question and copy the question's detiles to the relevant spaces, then choose what you whould like to predict and submit. the prediction result will appear on the right side of the screen 😉"
)

demo.launch()
