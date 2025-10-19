# Fake News Detector

A beginner-friendly machine learning system that classifies news headlines as real or fake using Natural Language Processing and Scikit-learn.

## Features

- Text Preprocessing: Stopword removal, stemming, and text cleaning
- Feature Extraction: TF-IDF and Count Vectorizer for converting text to numbers
- Multiple Models: Logistic Regression and Naive Bayes classifiers
- Model Evaluation: Accuracy scores, confusion matrices, and classification reports
- Easy Testing: Simple pipeline to test new news headlines

## Project Structure
fake-news-detector/
├── fake_news_detector.py
├── requirements.txt
└── README.md

text

## Installation

1. Clone the repository:

git clone https://github.com/Fancy-nateku/fake-news-detector.git
cd fake-news-detector

2.Install dependencies:

bash
pip install -r requirements.txt

Usage
Run the complete system:

bash
python fake_news_detector.py

Expected Output:

Dataset loading and preprocessing

Model training (Logistic Regression & Naive Bayes)

Evaluation metrics and visualizations

Predictions on test examples

Results
With the sample dataset (16 samples):

Logistic Regression: 40% accuracy

Naive Bayes: 40% accuracy

Note: These results are from a small demonstration dataset. With larger datasets, accuracy improves significantly.

How It Works
Text Cleaning: Convert to lowercase, remove special characters and stopwords

Stemming: Reduce words to their root form

Feature Extraction: Convert text to numerical features using TF-IDF

Model Training: Train machine learning classifiers

Prediction: Classify new text as REAL or FAKE news

Models Used
Logistic Regression with TF-IDF features

Naive Bayes with Count Vectorizer features

Improving the System
For better performance:

Use larger datasets (thousands of samples)

Try advanced models like Random Forest

Use word embeddings (Word2Vec, BERT)

Add more text features

Contributing
Contributions are welcome! Feel free to:

Report bugs

Suggest new features

Submit pull requests

License
This project is open source and available under the MIT License.

Built with Python, Scikit-learn, and NLTK
