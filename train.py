import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_optimized_model():
    print("Starting Optimized Model Training & Evaluation Pipeline...")
    
    # 1. Directories setup
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("static", "images"), exist_ok=True)
    
    # 2. Load Dataset
    if not os.path.exists("Dataset.csv"):
        print("Error: Dataset.csv not found in current directory.")
        return False
        
    print("Loading dataset...")
    df = pd.read_csv("Dataset.csv")
    
    # Clean text column
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'] != ""]
    
    X = df['cleaned_text']
    y = df['label']
    
    # 3. Train Test Split (80/20)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    # Save Vectorizer atomically
    tmp_vect_models = "models/tfidf_vectorizer.pkl.tmp"
    tmp_vect_root = "tfidf_vectorizer.pkl.tmp"
    with open(tmp_vect_models, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(tmp_vect_root, "wb") as f:
        pickle.dump(vectorizer, f)
    os.replace(tmp_vect_models, "models/tfidf_vectorizer.pkl")
    os.replace(tmp_vect_root, "tfidf_vectorizer.pkl")
    print("TF-IDF Vectorizer saved.")
    
    # 5. Define single optimized classifier (Logistic Regression with class balancing)
    print("Training Logistic Regression (Balanced class weights)...")
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_vect, y_train)
    
    # Save Model atomically
    tmp_model_path = "models/logistic_regression.pkl.tmp"
    with open(tmp_model_path, "wb") as f:
        pickle.dump(clf, f)
    os.replace(tmp_model_path, "models/logistic_regression.pkl")
    
    # Legacy file sync for legacy compatibility
    tmp_legacy_path = "cyber_model.pkl.tmp"
    with open(tmp_legacy_path, "wb") as f:
        pickle.dump(clf, f)
    os.replace(tmp_legacy_path, "cyber_model.pkl")
    print("Logistic Regression model saved.")
    
    # 6. Evaluate Model
    y_pred = clf.predict(X_test_vect)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"Results - Accuracy: {round(acc*100, 2)}%, F1-Score: {round(f1*100, 2)}%")
    
    # 7. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=['not_bullying', 'bullying'])
    plt.figure(figsize=(6, 5))
    # Customize plot color palette to fit corporate slate theme (Greys/Blues)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe', 'Bullying'],
                yticklabels=['Safe', 'Bullying'])
    plt.title('Confusion Matrix: Optimized Logistic Regression')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("static/images/confusion_logistic_regression.png", dpi=150)
    plt.close()
    
    # 8. Feature Importance Coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    classes = clf.classes_
    
    # Bullying coefficients
    if classes[1] == 'bullying':
        coef_bullying = coefs
        coef_safe = -coefs
    else:
        coef_bullying = -coefs
        coef_safe = coefs
        
    top_bullying_indices = np.argsort(coef_bullying)[-15:][::-1]
    top_bullying_words = [
        {"word": feature_names[i], "score": round(float(coef_bullying[i]), 4)}
        for i in top_bullying_indices
    ]
    
    top_safe_indices = np.argsort(coef_safe)[-15:][::-1]
    top_safe_words = [
        {"word": feature_names[i], "score": round(float(coef_safe[i]), 4)}
        for i in top_safe_indices
    ]
    
    metrics_summary = {
        "logistic_regression": {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "top_indicators": {
                "bullying": top_bullying_words,
                "safe": top_safe_words
            }
        }
    }
    
    # Save Metrics JSON
    with open("static/model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("Training process finished.")
    return True

if __name__ == "__main__":
    train_optimized_model()
