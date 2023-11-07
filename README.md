# Sentiment-Analysis-for-marketing

# INTRODUCTION:
Sentiment analysis, a powerful tool in modern marketing, involves the systematic analysis of public opinion, emotions, and attitudes expressed by consumers on various platforms, such as social media, reviews, and surveys. By harnessing natural language processing and machine learning techniques, marketers can gain valuable insights into customer sentiment.  

# OBJECTIVE:
To leverage sentiment analysis as a powerful tool for understanding and interpreting customer opinions, emotions, and feedback in order to enhance marketing strategies, improve customer satisfaction, and drive business growth.

# IMPORT THE NECESSARY LIBRARIES:
By Importing the Libraries like:
i)   NLTK (Natural Language Toolkit)
ii)  TextBlob
iii) VADER Sentiment
iv) Scikit-learn
v)  Pandas
vi) Matplotlib or Seaborn (for visualization)
vii) Wordcloud (optional, for word frequency visualizations)

# DATA SOURCE USED:
Using  Dataset:
(https://www.kaggle.com/datasets/crowdflower/twitter-
airline-sentiment)

# DATA COLLECTION:
Identify the sources of data: Determine where you will collect textual data for sentiment analysis. Common sources include customer reviews, social media platforms, surveys, emails, chat logs, and online forums.
Filepath = ‘../input/sentiment-analysis-for-marketing /tweets.csv’
		df = pd.read_csv(Filepath)
To ensure easy visual we are creating and exploration bar graph to see the analysis for marketing

# Load your dataset
data=pd.read_csv("/kaggle/input/sentiment-analysis-for-marketing/tweets.csv")

# EXPLORATORY DATA ANALYSIS(EDA):
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
print(BLUE + "\nDATA CLEANING" + RESET)
missing_values = df.isnull().sum()
print(GREEN + "Missing Values : " + RESET)
print(missing_values)

# REMOVING DUPLICATE VALUE:
mean_fill = df.fillna(df.mean())
df.fillna(mean_fill, inplace=True)
duplicate_values = df.duplicated().sum()
print(GREEN + "Duplicate Values : " + RESET)
print(duplicate_values)
df.drop_duplicates(inplace=True)

# Import necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Clean the Data:
Steps to clean the data for AI modeling.
print(BLUE + "\nDATA CLEANING" + RESET)

# DATA ANALYSIS:
print(BLUE + "\nDATA ANALYSIS" + RESET)
summary_stats = df.describe()
print(GREEN + "Summary Statistics : " + RESET)
print(summary_stats)

# MODEL ACCURACY:
print(BLUE + "\nMODELLING" + RESET);
X = df.drop("Outcome", axis=1);
y = df["Outcome"];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 769);
scaler = StandardScaler();
X_train = scaler.fit_transform(X_train);
X_test = scaler.transform(X_test);
model = svm.SVC(kernel="linear");
model.fit(X_train, y_train);
y_pred = model.predict(X_test);
accuracy = model.score(X_test, y_test);
print(GREEN + "Model Accuracy : " + RESET);
print(accuracy);

 # Data distribution
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Add labels to the bars
for i, count in enumerate(sentiment_counts.values):
plt.text(i, count + 10, str(count), ha='center', va='bottom')
plt.show()

# RESULT:
Display the sample result or performance metrices or visualization analysis obtained by using this procedure for sentiment for analysis marketing
