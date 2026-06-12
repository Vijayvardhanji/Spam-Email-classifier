"""
train.py
--------
Train and evaluate Naive Bayes and SVM classifiers on the
Spam Email dataset using TF-IDF features.

Run:
    python src/train.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving figures

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import load_and_preprocess, get_word_frequencies
from src.utils import (
    compute_metrics, save_model,
    plot_class_distribution, plot_word_frequency,
    plot_confusion_matrix, plot_accuracy_comparison
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'spam.csv')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
SHOTS_DIR    = os.path.join(BASE_DIR, 'screenshots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SHOTS_DIR,  exist_ok=True)


# ── TF-IDF explanation ───────────────────────────────────────────────────────
TFIDF_EXPLANATION = """
Why TF-IDF?
-----------
TF-IDF (Term Frequency – Inverse Document Frequency) is chosen because:

  1. Handles high-dimensionality: vocabulary can be 10k+ words; TF-IDF
     efficiently encodes them as sparse feature vectors.

  2. Down-weights common words: words like 'the', 'is' appear everywhere
     and carry little discriminative signal. IDF penalises them.

  3. Up-weights rare-but-discriminative words: e.g. 'prize', 'winner',
     'FREE' are rare overall but frequent in spam → high TF-IDF score.

  4. Works natively with Naive Bayes (non-negative values) and SVMs
     (well-scaled sparse vectors with linear kernel).

  5. Fast and memory-efficient compared to dense embeddings for this task.
"""


def main():
    print("=" * 60)
    print("   Spam Email Classifier – Training Pipeline")
    print("=" * 60)
    print(TFIDF_EXPLANATION)

    # ── 1. Load & preprocess ─────────────────────────────────────────────
    df = load_and_preprocess(DATASET_PATH)

    # ── 2. Visualise dataset distribution ───────────────────────────────
    plot_class_distribution(
        df,
        save_path=os.path.join(SHOTS_DIR, 'class_distribution.png')
    )

    # ── 3. Word-frequency charts ─────────────────────────────────────────
    ham_freq  = get_word_frequencies(df, 'ham',  top_n=15)
    spam_freq = get_word_frequencies(df, 'spam', top_n=15)
    plot_word_frequency(
        ham_freq, spam_freq,
        save_path=os.path.join(SHOTS_DIR, 'word_frequency.png')
    )

    # ── 4. Train / Test split ────────────────────────────────────────────
    X = df['clean_text']
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train size : {len(X_train)}")
    print(f"[INFO] Test  size : {len(X_test)}")

    # ── 5. Build pipelines ───────────────────────────────────────────────
    # TF-IDF parameters are shared for fair comparison
    tfidf_params = dict(
        max_features=5000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # apply log(1+tf) dampening
        min_df=2                 # ignore very rare terms
    )

    # Naive Bayes Pipeline
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   MultinomialNB(alpha=0.1))
    ])

    # SVM Pipeline  (LinearSVC wrapped in CalibratedClassifierCV for probabilities)
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf',   CalibratedClassifierCV(
                      LinearSVC(C=1.0, max_iter=2000, random_state=42)
                  ))
    ])

    # ── 6. Train ─────────────────────────────────────────────────────────
    print("\n[INFO] Training Naive Bayes ...")
    nb_pipeline.fit(X_train, y_train)

    print("[INFO] Training SVM ...")
    svm_pipeline.fit(X_train, y_train)

    # ── 7. Evaluate ──────────────────────────────────────────────────────
    nb_pred  = nb_pipeline.predict(X_test)
    svm_pred = svm_pipeline.predict(X_test)

    nb_metrics  = compute_metrics(y_test, nb_pred,  "Naive Bayes")
    svm_metrics = compute_metrics(y_test, svm_pred, "SVM (LinearSVC)")

    # Cross-validation scores
    print("\n[INFO] 5-Fold Cross-Validation Scores:")
    for name, pipe in [("Naive Bayes", nb_pipeline), ("SVM", svm_pipeline)]:
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
        print(f"  {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 8. Confusion matrices ────────────────────────────────────────────
    plot_confusion_matrix(
        y_test, nb_pred,  "Naive Bayes",
        save_path=os.path.join(SHOTS_DIR, 'cm_naive_bayes.png')
    )
    plot_confusion_matrix(
        y_test, svm_pred, "SVM (LinearSVC)",
        save_path=os.path.join(SHOTS_DIR, 'cm_svm.png')
    )

    # ── 9. Accuracy comparison ───────────────────────────────────────────
    plot_accuracy_comparison(
        [nb_metrics, svm_metrics],
        save_path=os.path.join(SHOTS_DIR, 'model_comparison.png')
    )

    # ── 10. Save models ──────────────────────────────────────────────────
    save_model(nb_pipeline,  os.path.join(MODELS_DIR, 'naive_bayes.pkl'))
    save_model(svm_pipeline, os.path.join(MODELS_DIR, 'svm.pkl'))

    print("\n[✓] Training complete. Models saved to /models/")
    print("[✓] Visualisations saved to /screenshots/")
    print("\nSummary")
    print("-" * 40)
    print(f"  Naive Bayes  Accuracy : {nb_metrics['accuracy']:.4f}")
    print(f"  SVM          Accuracy : {svm_metrics['accuracy']:.4f}")
    winner = "SVM" if svm_metrics['f1'] > nb_metrics['f1'] else "Naive Bayes"
    print(f"\n  Best F1-Score: {winner}")


if __name__ == '__main__':
    main()
