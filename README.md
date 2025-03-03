# MACHINE-LEARNING-MODEL-IMPLEMENTATION


_COMPANY:_ CODTECH IT SOLUTIONS

_NAME:_ ARMAN KHAN

_INTERN ID:_ CT08RRC

_DOMAIN:_ PYTHON 

_DURATION:_ 4 WEEKS

_MENTOR:_  NEELA SANTOSH

## DESCRIPTION OF TAKS
This code provides a simple pipeline for spam classification using machine learning, specifically with a Naive Bayes classifier. The goal is to categorize text messages as "ham" (non-spam) or "spam." The process begins by importing essential libraries like `pandas` for data manipulation, `train_test_split` from `sklearn.model_selection` to split the data into training and testing sets, `TfidfVectorizer` from `sklearn.feature_extraction.text` to convert text into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF), and `MultinomialNB` from `sklearn.naive_bayes` for building the model. Additionally, `classification_report` and `accuracy_score` from `sklearn.metrics` are imported to evaluate model performance.

The code then loads the dataset from a CSV file (`mail_data.csv`), which contains labeled messages categorized as either "ham" or "spam." This dataset is read into a `pandas` DataFrame, allowing for easy manipulation. During preprocessing, the target variable `Category` is mapped from categorical labels ("ham" and "spam") to numerical values (0 for "ham" and 1 for "spam"), making it suitable for machine learning models.

Next, the data is divided into features (`Message` column) and labels (`Category` column). The `train_test_split()` function is used to split the data into training and testing sets, with 80% of the data used for training and 20% for testing, ensuring that the model can be evaluated on unseen data.

To handle the text data, the `TfidfVectorizer` is used to transform the text into numerical features, giving a weight to each word based on its frequency in a document and its relative importance across the dataset. The training data is transformed using `fit_transform()`, and the test data is transformed using `transform()` to maintain consistency.

After transforming the text data, the Naive Bayes model is trained using the `MultinomialNB` classifier. This probabilistic model assumes features (words) are conditionally independent given the class label, making it suitable for text classification tasks. The model is trained using the `fit()` method on the transformed training data and labels. The trained model is then used to predict labels (spam or non-spam) for the test set using the `predict()` method.

Finally, the model’s performance is evaluated using `accuracy_score` to measure the overall accuracy and `classification_report` for a detailed evaluation, including precision, recall, and F1-score. In conclusion, this code offers a simple yet effective approach for spam detection using Naive Bayes, leveraging TF-IDF for feature extraction and evaluating the model with common metrics.


## OUTPUT PICTURE
![Image](https://github.com/user-attachments/assets/a9547ab7-fa1d-4b78-a0de-0d30e8084c7f)
