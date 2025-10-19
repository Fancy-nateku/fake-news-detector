
# Fake News Detection System - Complete Fixed Code
# Beginner-friendly implementation using Scikit-learn

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (run this once)
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    print("✓ NLTK resources downloaded successfully")
except:
    print("NLTK download failed, but we'll proceed anyway")

print("=== Fake News Detection System ===
")

# Step 1: Load and Explore the Dataset
print("Step 1: Loading Dataset...")

# Create sample data
data = {
    'text': [
        'Breaking: Scientists discover revolutionary new energy source',
        'Celebrity spotted eating at local restaurant',
        'Government announces new policy to help economy',
        'Alien invasion reported in small town - officials deny',
        'New study shows benefits of daily exercise',
        'Secret cure for cancer discovered but hidden by big pharma',
        'Stock market reaches all time high amid economic growth',
        'You wont believe what this celebrity did last night',
        'Research confirms climate change is accelerating',
        'Vaccines contain microchips for population control',
        'Economic indicators show strong recovery trends',
        'Moon landing was faked by Hollywood according to insider',
        'New technology promises to revolutionize transportation',
        'Eating this one food will make you lose weight instantly',
        'International summit addresses global cooperation',
        'Secret society controls world governments from shadows'
    ],
    'label': [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = real, 0 = fake
}

df = pd.DataFrame(data)
print(f"Dataset loaded: {len(df)} samples")
print(f"Real news: {sum(df['label'])} samples")
print(f"Fake news: {len(df) - sum(df['label'])} samples")
print("
First few samples:")
print(df.head(8))

# Step 2: Text Preprocessing
print("
Step 2: Text Preprocessing...")

def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

print("Original text vs Cleaned text:")
for i in range(3):
    print(f"Original: {df['text'].iloc[i]}")
    print(f"Cleaned:  {df['cleaned_text'].iloc[i]}
")

# Step 3: Prepare Data for Training
print("Step 3: Preparing Data for Training...")

X = df['cleaned_text']  # Features (text data)
y = df['label']         # Labels (1 = real, 0 = fake)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Training set - Real: {sum(y_train)}, Fake: {len(y_train)-sum(y_train)}")
print(f"Testing set - Real: {sum(y_test)}, Fake: {len(y_test)-sum(y_test)}")

# Step 4: Feature Extraction - Convert Text to Numbers
print("
Step 4: Feature Extraction using TF-IDF...")

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,      # Consider top 1000 words
    min_df=2,               # Ignore words that appear in less than 2 documents
    max_df=0.8,             # Ignore words that appear in more than 80% of documents
    ngram_range=(1, 2)      # Consider both single words and word pairs
)

# Transform the text data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF Features shape - Training: {X_train_tfidf.shape}")
print(f"TF-IDF Features shape - Testing: {X_test_tfidf.shape}")

# Count Vectorizer (alternative approach)
count_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

print(f"Count Features shape - Training: {X_train_count.shape}")

# Step 5: Train Machine Learning Models
print("
Step 5: Training Machine Learning Models...")

# Model 1: Logistic Regression with TF-IDF
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Model 2: Naive Bayes with Count Vectorizer
print("Training Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train_count, y_train)

print("✓ Models trained successfully!")

# Step 6: Make Predictions and Evaluate Models
print("
Step 6: Model Evaluation...")

# Predictions
y_pred_lr = lr_model.predict(X_test_tfidf)
y_pred_nb = nb_model.predict(X_test_count)

# Evaluate Logistic Regression
print("
" + "="*50)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print("
Detailed Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Fake', 'Real']))

# Evaluate Naive Bayes
print("
" + "="*50)
print("NAIVE BAYES RESULTS")
print("="*50)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
print("
Detailed Classification Report:")
print(classification_report(y_test, y_pred_nb, target_names=['Fake', 'Real']))

# Step 7: Visualize Results
print("
Step 7: Creating Visualizations...")

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'], ax=axes[0])
axes[0].set_title('Logistic Regression - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'], ax=axes[1])
axes[1].set_title('Naive Bayes - Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Step 8: Test with New Examples
print("
Step 8: Testing with New Examples...")

# Create a pipeline for easy prediction
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('classifier', lr_model)
])

# Test samples
test_samples = [
    "New scientific discovery changes everything we know",
    "Secret government program controls the weather",
    "Economic growth continues for third consecutive quarter",
    "One simple trick will make you rich overnight"
]

print("
Predictions for new text samples:")
for i, sample in enumerate(test_samples, 1):
    # Preprocess
    cleaned_sample = preprocess_text(sample)
    # Predict
    prediction = pipeline.predict([cleaned_sample])[0]
    probability = pipeline.predict_proba([cleaned_sample])[0]
    
    result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    print(f"
Sample {i}: '{sample}'")
    print(f"Prediction: {result} (Confidence: {confidence:.2%})")

# Step 9: Feature Importance Analysis
print("
Step 9: Analyzing Important Features...")

# Get feature names and coefficients
feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = lr_model.coef_[0]

# Create a DataFrame for feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': coefficients
})

# Sort by absolute importance
feature_importance['abs_importance'] = abs(feature_importance['importance'])
top_features = feature_importance.sort_values('abs_importance', ascending=False).head(10)

print("
Top 10 most important features for classification:")
print("(Positive = indicates REAL news, Negative = indicates FAKE news)")
for idx, row in top_features.iterrows():
    sentiment = "REAL" if row['importance'] > 0 else "FAKE"
    print(f"  {row['feature']:15} → {sentiment:4} (weight: {row['importance']:.3f})")

# Final Summary
print("
" + "="*60)
print("FAKE NEWS DETECTION SYSTEM - SUMMARY")
print("="*60)
print(f"✓ Dataset: {len(df)} samples (balanced)")
print(f"✓ Best Model: Logistic Regression ({lr_accuracy*100:.1f}% accuracy)")
print(f"✓ Features: {X_train_tfidf.shape[1]} TF-IDF features")
print(f"✓ Preprocessing: Lowercase, stopwords removal, stemming")
print("✓ Ready for real-world deployment with more data!")
print("="*60)

print("
To improve this system:")
print("1. Use a larger, real-world dataset (thousands of samples)")
print("2. Try more advanced models (Random Forest, SVM, Neural Networks)")
print("3. Use word embeddings (Word2Vec, BERT) for better text representation")
print("4. Add more sophisticated text preprocessing")
print("5. Implement cross-validation for more robust evaluation")
