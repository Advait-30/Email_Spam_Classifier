"""
Email Spam Classifier using Sentence Transformers and Naive Bayes
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import urllib.request

class EmailSpamClassifier:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the spam classifier with a specified sentence transformer model.
        
        Args:
            model_name (str): Name of the HuggingFace sentence-transformer model
        """
        self.model_name = model_name
        self.embedding_model = None
        self.classifier = None
        self.embeddings_cache = {}
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load the SMS Spam Collection dataset. If filepath is not provided,
        download from the UCI ML Repository.
        
        Args:
            filepath (str, optional): Path to local CSV file
            
        Returns:
            pd.DataFrame: DataFrame with 'text' and 'label' columns
        """
        if filepath is None:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
            print("Downloading dataset...")
            urllib.request.urlretrieve(url, "spam.zip")
            import zipfile
            with zipfile.ZipFile("spam.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            filepath = "SMSSpamCollection"
        
        # Read the data with appropriate column names
        df = pd.read_csv(filepath, sep='\t', names=['label', 'text'], encoding='utf-8')
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        return df

    def preprocess_text(self, text: str) -> str:
        """
        Clean text by converting to lowercase and removing special characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string using sentence transformer model.
        Uses caching to avoid recomputing embeddings.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Text embedding
        """
        if text not in self.embeddings_cache:
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(self.model_name)
            self.embeddings_cache[text] = self.embedding_model.encode(text)
        return self.embeddings_cache[text]

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess texts and split into train/test sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'text' and 'label' columns
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple containing train/test features and labels
        """
        # Preprocess all texts
        print("Preprocessing texts...")
        texts = [self.preprocess_text(text) for text in tqdm(df['text'])]
        
        # Generate embeddings
        print("Generating embeddings...")
        X = np.array([self.get_embedding(text) for text in tqdm(texts)])
        y = df['label'].values
        
        # Split data
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the Naive Bayes classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model using various metrics.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.classifier.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def predict_email(self, email_text: str) -> Tuple[str, float]:
        """
        Predict whether a given email is spam or ham.
        
        Args:
            email_text (str): Input email text
            
        Returns:
            Tuple[str, float]: Prediction ('spam' or 'ham') and confidence score
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess and embed the text
        processed_text = self.preprocess_text(email_text)
        embedding = self.get_embedding(processed_text)
        
        # Get prediction and probability
        pred = self.classifier.predict([embedding])[0]
        prob = self.classifier.predict_proba([embedding])[0]
        
        # Return prediction and confidence
        label = 'spam' if pred == 1 else 'ham'
        confidence = prob[1] if pred == 1 else prob[0]
        
        return label, confidence

def main():
    """Main function to demonstrate the spam classifier pipeline."""
    
    # Initialize classifier
    classifier = EmailSpamClassifier()
    
    # Load data
    print("Loading dataset...")
    df = classifier.load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Train model
    print("Training model...")
    classifier.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = classifier.evaluate(X_test, y_test)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Example predictions
    print("\nExample Predictions:")
    example_texts = [
        "URGENT! You have won a free vacation. Click here to claim now!",
        "Hi Mom, what time should I come over for dinner tonight?"
    ]
    
    for text in example_texts:
        label, confidence = classifier.predict_email(text)
        print(f"\nText: {text}")
        print(f"Prediction: {label} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
