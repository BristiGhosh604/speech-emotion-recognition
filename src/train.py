import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_cross_validation(X, y):
    """
    5-fold cross-validation: trains 5 separate models on different data splits,
    and reports mean +/- standard deviation accuracy -- a more honest estimate
    than a single train/test split.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"\nCross-validation scores (5 folds): {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

def main():
    # Load the features and labels we saved earlier
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")
    print(f"Loaded X={X.shape}, y={y.shape}")
    run_cross_validation(X, y)
    
    # Split: 80% train, 20% test. stratify=y ensures each emotion is
    # proportionally represented in both sets (not all "happy" ending up in test, etc.)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Scale features: Random Forest doesn't strictly need this, but it's good
    # practice, and it matters a lot once we compare against other models later.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest
    model = RandomForestClassifier(
        n_estimators=200,   # number of trees in the forest
        max_depth=15,       # how deep each tree can grow (limits overfitting)
        random_state=42,
        n_jobs=-1,           # use all CPU cores for speed
    )
    print("Training model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate on the held-out test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\nDetailed report (per emotion):")
    print(classification_report(y_test, y_pred))

    # Save the trained model + scaler so we can use them later without retraining
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/emotion_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print("Model saved to outputs/emotion_model.pkl")


if __name__ == "__main__":
    main()