import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
import spacy

# _______________________
# difficulty + is premium
# _______________________

# Load the CSV file
file_path = "data.csv"
df = pd.read_csv(file_path)

# Set plot style
sns.set_style("whitegrid")

# Create a histogram (count plot) of difficulty grouped by is_premium
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="difficulty", hue="is_premium", palette=["#A6CDC6", "#DDA853"])

# Labels and title
plt.xlabel("Difficulty Level")
plt.ylabel("Count")
plt.title("Distribution of Difficulty Levels by Premium Status")
plt.legend(title="Is Premium", labels=["Non-Premium", "Premium"])

# Show plot
plt.show()

# ____________________________
# difficulty + acceptance rate
# ____________________________

bins = range(0, 105, 5)  # Bins from 0 to 100 in steps of 5
labels = [f"{i}-{i+5}%" for i in bins[:-1]]

# Add a new column for acceptance rate bins
df["acceptance_rate_bin"] = pd.cut(df["acceptance_rate"], bins=bins, labels=labels, right=False)

# Plot the histogram (count plot) of difficulty grouped by acceptance rate bins
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="acceptance_rate_bin", hue="difficulty", palette=["#A6CDC6", "#DDA853", "#3B6790"])

# Labels and title
plt.xlabel("Acceptance Rate (%)")
plt.ylabel("Count")
plt.title("Distribution of Difficulty Levels by Acceptance Rate")
plt.xticks(rotation=45)
plt.legend(title="Difficulty")

# Show plot
plt.tight_layout()
plt.show()

# _____________________________________
# difficulty + acceptance rate separate
# _____________________________________

# Create bins for acceptance_rate in intervals of 5
bins = range(0, 105, 5)
labels = [f"{i}-{i+5}%" for i in bins[:-1]]

# Add a new column for acceptance rate bins
df["acceptance_rate_bin"] = pd.cut(df["acceptance_rate"], bins=bins, labels=labels, right=False)

# Define difficulty levels and colors
difficulty_levels = ["Easy", "Medium", "Hard"]
colors = ["#A6CDC6", "#DDA853", "#3B6790"]

# Create separate histograms for each difficulty level
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for i, difficulty in enumerate(difficulty_levels):
    sns.countplot(
        data=df[df["difficulty"] == difficulty],
        x="acceptance_rate_bin",
        color=colors[i],
        ax=axes[i]
    )
    axes[i].set_title(f"{difficulty} Problems")
    axes[i].set_ylabel("Count")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45)

# Set common x-axis label
axes[-1].set_xlabel("Acceptance Rate (%)")

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# ____________________________
# difficulty + discuss count
# ____________________________

bins = range(0, 1050, 50)  # Bins from 0 to 1000 in steps of 5
labels = [f"{i}-{i+50}" for i in bins[:-1]]

# Add a new column for acceptance rate bins
df["discuss_count_bin"] = pd.cut(df["discuss_count"], bins=bins, labels=labels, right=False)

# Plot the histogram (count plot) of difficulty grouped by acceptance rate bins
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="discuss_count_bin", hue="difficulty", palette=["#A6CDC6", "#DDA853", "#3B6790"])

# Labels and title
plt.xlabel("Discuss Count")
plt.ylabel("Count")
plt.title("Distribution of Difficulty Levels by Discuss Count")
plt.xticks(rotation=45)
plt.legend(title="Difficulty")

# Show plot
plt.tight_layout()
plt.show()

# ___________________________________
# difficulty + discuss count separate
# ___________________________________

# Create bins for discuss_count in intervals of 5
bins = range(0, 1050, 50)
labels = [f"{i}-{i+50}" for i in bins[:-1]]

# Add a new column for acceptance rate bins
df["discuss_count_bin"] = pd.cut(df["discuss_count"], bins=bins, labels=labels, right=False)

# Define difficulty levels and colors
difficulty_levels = ["Easy", "Medium", "Hard"]
colors = ["#A6CDC6", "#DDA853", "#3B6790"]

# Create separate histograms for each difficulty level
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for i, difficulty in enumerate(difficulty_levels):
    sns.countplot(
        data=df[df["difficulty"] == difficulty],
        x="discuss_count_bin",
        color=colors[i],
        ax=axes[i]
    )
    axes[i].set_title(f"{difficulty} Problems")
    axes[i].set_ylabel("Count")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45)

# Set common x-axis label
axes[-1].set_xlabel("Discuss Count")

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# ____________________
# premium + difficulty
# ____________________

# Create the crosstab table
table = pd.crosstab(df["is_premium"], df["difficulty"], margins=True, margins_name="Total")

# Convert the table to NumPy array for plotting
table_data = table.values
columns = ["Is Premium"] + table.columns.tolist()  # Corrected column labels
rows = table.index.tolist()

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Prepare table data with matching columns
table_text = [columns] + [[str(rows[i])] + list(map(str, table_data[i])) for i in range(len(rows))]

# Define cell colors: first row & first column colored
cell_colors = [["lightgray"] * len(columns)] + [["lightgray"] + ["white"] * (len(columns) - 1)
                                                for _ in range(len(rows))]

# Create table
table_plot = ax.table(cellText=table_text, cellLoc="center", loc="center", cellColours=cell_colors)

# Adjust table appearance
table_plot.scale(1.2, 1.2)  # Adjust size
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(10)

plt.title("Is Premium vs. Difficulty Table", fontsize=14, fontweight="bold")
plt.show()

# ______________________
# companies + difficulty
# ______________________

# Split the 'companies' column and explode it into separate rows
df["companies"] = df["companies"].str.split(",")
df = df.explode("companies")

# Get the 20 most common companies
top_companies = df["companies"].value_counts().nlargest(20).index

# Filter the dataframe to keep only the top 20 companies
df = df[df["companies"].isin(top_companies)]

# Group by company and difficulty, then count occurrences
company_difficulty_counts = df.groupby(["companies", "difficulty"]).size().unstack(fill_value=0)

# Plot the histogram as a stacked bar chart
palette = ["#A6CDC6", "#3B6790", "#DDA853"]
company_difficulty_counts.plot(kind="bar", color=palette, figsize=(12, 6))

# Labels and title
plt.xlabel("Company")
plt.ylabel("Count of Problems")
plt.title("Problem Difficulty Distribution by Company (Top 20)")
plt.xticks(rotation=45)
plt.legend(title="Difficulty")

# Show plot
plt.tight_layout()
plt.show()

# ___________________
# topics + difficulty
# ___________________

# Split the 'related_topics' column and explode it into separate rows
df["related_topics"] = df["related_topics"].str.split(",")
df = df.explode("related_topics")

# Get the 30 most common companies
top_related_topics = df["related_topics"].value_counts().nlargest(30).index

# Filter the dataframe
df = df[df["related_topics"].isin(top_related_topics)]

# Group by company and difficulty, then count occurrences
related_topics_difficulty_counts = df.groupby(["related_topics", "difficulty"]).size().unstack(fill_value=0)

# Plot the histogram as a stacked bar chart
palette = ["#A6CDC6", "#3B6790", "#DDA853"]
related_topics_difficulty_counts.plot(kind="bar", color=palette, figsize=(12, 6), width=0.9)

# Labels and title
plt.xlabel("Related Topics")
plt.ylabel("Count of Problems")
plt.title("Problem Difficulty Distribution by Related Topics")
plt.xticks(rotation=90)
plt.legend(title="Difficulty")

# Show plot
plt.tight_layout()
plt.show()

# __________________________________
# companies + related topics (ratio)
# __________________________________

# Drop rows where either 'companies' or 'related_topics' is missing
df_filtered = df.dropna(subset=['companies', 'related_topics'])

# Initialize dictionary to store frequency of topics per company
company_topic_freq = defaultdict(lambda: defaultdict(int))

# Iterate through the dataset
for _, row in df_filtered.iterrows():
    companies = row['companies'].split(',')
    topics = row['related_topics'].split(',')

    for company in companies:
        for topic in topics:
            company_topic_freq[company.strip()][topic.strip()] += 1

# Convert to a DataFrame
company_topic_df = pd.DataFrame(company_topic_freq).fillna(0).astype(int).T

# Show the first few rows of the result
print(company_topic_df.head())

# Normalize each company's topic frequency by dividing by the total questions they are associated with
company_topic_ratio_df = company_topic_df.div(company_topic_df.sum(axis=1), axis=0)

# Select top companies for clarity
filtered_ratio_df = company_topic_ratio_df.loc[top_companies]

# Set plot size
plt.figure(figsize=(12, 8))

# Create heatmap with ratio values
sns.heatmap(filtered_ratio_df, cmap="Blues", linewidths=0.5)

# Labels and title
plt.title("Ratio of Related Topics Across Top Companies", fontsize=14)
plt.xlabel("Related Topics", fontsize=12)
plt.ylabel("Companies", fontsize=12)

# Show plot
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

# _____________________________________________________________________________________________
# most common words in the descriptions and how many times they appear in each difficulty level
# _____________________________________________________________________________________________

nlp = spacy.load("en_core_web_sm")

# Ensure 'description' and 'difficulty' columns exist
if "description" not in df.columns or "difficulty" not in df.columns:
    raise ValueError("The CSV file must contain 'description' and 'difficulty' columns.")

# Extract descriptions and difficulties
descriptions = df["description"]
difficulties = df["difficulty"]

# Define manual word mapping (add more as needed)
word_mapping = {
    "arr": "array",
    "nums": "num",
    "num": "number",
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "node": "node"
}


# Function for manual word mapping after lemmatization
def apply_word_mapping(text):
    words = text.split()
    mapped_words = [word_mapping.get(word, word) for word in words]  # Replace with mapped word if exists
    return " ".join(mapped_words)


# Function for lemmatizing words in the description text
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return " ".join(lemmatized_words)


# Apply lemmatization first
lemmatized_descriptions = descriptions.apply(lemmatize_text)

# Apply the manual word mapping after lemmatization
mapped_lemmatized_descriptions = lemmatized_descriptions.apply(apply_word_mapping)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",  # Removes common English words
    max_features=50,  # Select top 50 words
    min_df=2,  # Ignore words appearing in <2 documents
    max_df=0.8,  # Ignore words appearing in >80% of documents
)

# Fit and transform the TF-IDF Vectorizer on the processed descriptions
tfidf_matrix = vectorizer.fit_transform(mapped_lemmatized_descriptions)

# Get the top words from TF-IDF
top_words = vectorizer.get_feature_names_out()
top_words = np.delete(top_words, np.where(top_words == "nums"))
top_words = np.delete(top_words, np.where(top_words == "arr"))

print(f"Top words from TF-IDF: {top_words}")


# Create a Counter for each difficulty level
word_counts = {"Easy": Counter(), "Medium": Counter(), "Hard": Counter()}

# Iterate through each description and its corresponding difficulty
for idx, desc in enumerate(mapped_lemmatized_descriptions):
    difficulty = difficulties.iloc[idx]  # Get the difficulty for this question
    if difficulty not in word_counts:
        continue  # Skip if difficulty is not recognized (i.e., not "easy", "medium", or "hard")

    # Get the words in the description (case-insensitive matching)
    words_in_desc = set(desc.lower().split())  # Split description into words (convert to lowercase)

    # Count the words that appear in the description
    for word in top_words:
        if word in words_in_desc:
            word_counts[difficulty][word] += 1

# Prepare data for plotting
easy_counts = [word_counts["Easy"][word] for word in top_words]
medium_counts = [word_counts["Medium"][word] for word in top_words]
hard_counts = [word_counts["Hard"][word] for word in top_words]

# Plot the histogram
x = np.arange(len(top_words))

plt.figure(figsize=(14, 8))  # Increase figure size for better spacing
bar_width = 0.25  # Set the bar width smaller to avoid overlap

plt.bar(x - bar_width, easy_counts, bar_width, label="Easy", color='#A6CDC6')
plt.bar(x, medium_counts, bar_width, label="Medium", color='#DDA853')
plt.bar(x + bar_width, hard_counts, bar_width, label="Hard", color='#3B6790')

plt.xlabel("Words")
plt.ylabel("Count")
plt.title("Word Counts by Difficulty Level")
plt.xticks(x, top_words, rotation=90)
plt.legend()

# Adjust layout to ensure proper spacing
plt.tight_layout()
plt.show()
