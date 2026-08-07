import sys
import os
import numpy as np

# Let this script import from src/, since it lives in scripts/ instead
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import load_dataset
from features import extract_features


def main():
    print("Finding audio files...")
    samples = load_dataset("data/raw")
    print(f"Found {len(samples)} files")

    features_list = []
    labels_list = []

    for i, (file_path, label) in enumerate(samples):
        if i % 100 == 0:
            print(f"Processing {i}/{len(samples)}...")

        try:
            features = extract_features(file_path)
            features_list.append(features)
            labels_list.append(label)
        except Exception as e:
            print(f"  Skipped {file_path}: {e}")

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)
    print("Saved to data/processed/X.npy and data/processed/y.npy")


if __name__ == "__main__":
    main()