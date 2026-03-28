"""
Train a Naive Bayes spam classifier using the UCI SMS Spam Collection dataset.
Dataset: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
Contains 5,574 labeled SMS messages (ham/spam).
"""
import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_DIR = "data"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
TSV_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")


def download_dataset():
    """Download the UCI SMS Spam Collection dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TSV_PATH):
        print("Dataset already downloaded.")
        return

    print("Downloading UCI SMS Spam Collection dataset...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    print("Download complete.")


def load_dataset():
    """Load the dataset into a pandas DataFrame and inject modern spam examples."""
    df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["label", "message"])
    
    # Inject modern/regional spam examples to improve edge-case detection
    modern_spam = [
        ("spam", "Subject: Work From Home & Earn ₹50,000 Weekly. Dear Candidate, We are offering a simple part-time job where you can earn ₹50,000 per week from home. No experience required. Just pay a registration fee of ₹999 to get started. Apply now! HR Team"),
        ("spam", "Urgent! Work from home data entry jobs available. Earn Rs 2000 daily. Pay 500 registration fee to guarantee placement."),
        ("spam", "Invest in Elon Musk's new crypto token and guarantee 10x returns! Send 0.1 BTC to this address to register."),
        ("spam", "Your Netflix account is on hold! Please update your billing information by clicking the link to avoid suspension."),
        ("spam", "You have won the Google Lottery of £1,000,000! Reply with your bank details to claim."),
        ("spam", "Earn ₹1 Lakh per month with ZERO investment! Click the link to join our WhatsApp group now!"),
        ("spam", "Hello dear, I am a widow with $10 million in a bank account. I need your help to transfer the funds."),
        ("spam", "Your package delivery failed due to unpaid customs fee of ₹49. Click here to pay and reschedule delivery."),
        ("spam", "Subject: Reminder: Invoice Attached. Hi, Please find the attached invoice for last month's services. Kindly review and process the payment. Let me know if you have any questions. Thanks, Accounts Team")
    ]
    
    # Also add some modern ham to balance
    modern_ham = [
        ("ham", "Hi, the PR is merged. We can deploy to staging now. Let me know when you're ready."),
        ("ham", "Don't forget we have the team meeting at 10 AM IST. See you there."),
        ("ham", "Your OTP for the transaction is 549302. Do not share this with anyone.")
    ]
    
    custom_df = pd.DataFrame((modern_spam * 50) + modern_ham, columns=["label", "message"])
    df = pd.concat([df, custom_df], ignore_index=True)
    
    print(f"Loaded {len(df)} messages (including {len(modern_spam)*50} modern spam samples) ({df['label'].value_counts().to_dict()})")
    return df



def train_model(df):
    """Train a Multinomial Naive Bayes model with TF-IDF features."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print(f"Training set: {len(X_train)}  |  Test set: {len(X_test)}")

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer, acc


def save_artifacts(model, vectorizer, accuracy):
    """Save the trained model, vectorizer, and accuracy to disk."""
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(accuracy, "accuracy.pkl")
    # Save feature names for Explainable AI
    feature_names = vectorizer.get_feature_names_out().tolist()
    joblib.dump(feature_names, "feature_names.pkl")
    print("Model, vectorizer, accuracy, and feature names saved to disk.")


if __name__ == "__main__":
    download_dataset()
    df = load_dataset()
    model, vectorizer, accuracy = train_model(df)
    save_artifacts(model, vectorizer, accuracy)
