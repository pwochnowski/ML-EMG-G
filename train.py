"""Train and evaluate classifiers on pre-extracted EMG features (.npz).

Usage examples:
  uv run train.py --data /tmp/s1_pos1_test.npz
  uv run train.py --data S1_Male_features.npz --models lda,svm --test-size 0.2

The script trains baseline classifiers (LDA and SVM by default), prints
classification reports and a confusion matrix, and can save the best model.
"""

from pathlib import Path
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from models import build_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    return X, y


# build_model is provided by the shared `models` module


def evaluate_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm


def parse_args():
    p = argparse.ArgumentParser(description='Train classifiers on EMG features (.npz)')
    p.add_argument('--data', required=True, help='Path to .npz feature file')
    p.add_argument('--models', default='lda,svm', help='Comma-separated models to train (lda,svm)')
    p.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--save-model', help='Path to save best model (joblib)')
    p.add_argument('--cv', type=int, default=0, help='If >0, run CV with this many folds instead of a train/test split')
    return p.parse_args()


def main():
    args = parse_args()
    X, y = load_data(args.data)

    models = [m.strip() for m in args.models.split(',') if m.strip()]

    if args.cv and args.cv > 1:
        print(f"Running cross-validation (k={args.cv}) for models: {models}")
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        for m in models:
            pipe =build_model(m)
            scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
            print(f"Model: {m}  CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}  (n={len(scores)})")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results = []
    for m in models:
        print(f"\nTraining model: {m}")
        pipe = build_model(m)
        acc, report, cm = evaluate_model(pipe, X_train, X_test, y_train, y_test)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report:\n", report)
        print("Confusion matrix:\n", cm)
        results.append((m, acc, pipe))

    # save best model if requested
    if args.save_model:
        best = max(results, key=lambda r: r[1])
        save_path = Path(args.save_model)
        joblib.dump(best[2], save_path)
        print(f"Saved best model ({best[0]}, acc={best[1]:.4f}) to: {save_path}")


if __name__ == '__main__':
    main()
