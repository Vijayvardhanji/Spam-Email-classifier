"""
predict.py
----------
Prediction interface for the Spam Email Classifier.
Loads saved models and returns predictions with confidence scores.

Usage (CLI):
    python src/predict.py "Congratulations! You've won a FREE prize. Call now!"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import clean_text
from src.utils import load_model

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

NB_MODEL_PATH  = os.path.join(MODELS_DIR, 'naive_bayes.pkl')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm.pkl')


class SpamPredictor:
    """
    Wrapper class that loads both trained pipelines and exposes
    a simple predict() interface.
    """

    def __init__(self, model_name: str = 'svm'):
        """
        Args:
            model_name (str): 'svm' or 'naive_bayes'
        """
        self.model_name = model_name.lower()
        self._pipeline = None

    def _ensure_loaded(self):
        """Lazy-load model on first prediction."""
        if self._pipeline is not None:
            return

        path_map = {
            'svm':         SVM_MODEL_PATH,
            'naive_bayes': NB_MODEL_PATH,
            'nb':          NB_MODEL_PATH,
        }
        path = path_map.get(self.model_name)
        if path is None:
            raise ValueError(f"Unknown model: {self.model_name}. "
                             f"Choose from {list(path_map.keys())}")
        self._pipeline = load_model(path)

    def predict(self, text: str) -> dict:
        """
        Predict whether a single email / SMS is spam or ham.

        Args:
            text (str): Raw email text.

        Returns:
            dict:
                label        → 'Spam' | 'Ham'
                label_id     → 1 | 0
                confidence   → float in [0, 1]
                spam_prob    → probability of being spam
                ham_prob     → probability of being ham
                clean_text   → preprocessed text used for prediction
        """
        self._ensure_loaded()

        # Preprocess
        cleaned = clean_text(text)

        # Predict
        label_id   = int(self._pipeline.predict([cleaned])[0])
        proba      = self._pipeline.predict_proba([cleaned])[0]
        ham_prob   = float(proba[0])
        spam_prob  = float(proba[1])
        confidence = float(max(proba))

        return {
            'label':      'Spam' if label_id == 1 else 'Ham',
            'label_id':   label_id,
            'confidence': confidence,
            'spam_prob':  spam_prob,
            'ham_prob':   ham_prob,
            'clean_text': cleaned,
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict for a list of texts.

        Args:
            texts (list[str]): List of raw email strings.

        Returns:
            list[dict]: Prediction dicts (same structure as predict()).
        """
        return [self.predict(t) for t in texts]


def predict_both_models(text: str) -> dict:
    """
    Run prediction through both Naive Bayes and SVM and return
    a combined result dict – handy for comparison.

    Args:
        text (str): Raw email text.

    Returns:
        dict: {'naive_bayes': {...}, 'svm': {...}}
    """
    nb_predictor  = SpamPredictor('naive_bayes')
    svm_predictor = SpamPredictor('svm')

    return {
        'naive_bayes': nb_predictor.predict(text),
        'svm':         svm_predictor.predict(text),
    }


def _cli():
    """Command-line interface for quick predictions."""
    parser = argparse.ArgumentParser(
        description='Spam Email Classifier – Predict'
    )
    parser.add_argument('text', type=str,
                        help='Email text to classify')
    parser.add_argument('--model', type=str, default='svm',
                        choices=['svm', 'naive_bayes'],
                        help='Model to use (default: svm)')
    parser.add_argument('--both', action='store_true',
                        help='Run both models and compare')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  Spam Email Classifier – Prediction")
    print("="*55)
    print(f"  Input: {args.text[:80]}{'...' if len(args.text)>80 else ''}")
    print("="*55)

    if args.both:
        results = predict_both_models(args.text)
        for mname, res in results.items():
            icon = "🚫" if res['label'] == 'Spam' else "✅"
            print(f"\n  [{mname.upper()}]")
            print(f"  {icon} Prediction : {res['label']}")
            print(f"     Confidence  : {res['confidence']*100:.1f}%")
            print(f"     Spam Prob   : {res['spam_prob']*100:.1f}%")
            print(f"     Ham Prob    : {res['ham_prob']*100:.1f}%")
    else:
        predictor = SpamPredictor(args.model)
        res = predictor.predict(args.text)
        icon = "🚫" if res['label'] == 'Spam' else "✅"
        print(f"\n  {icon} Prediction : {res['label']}")
        print(f"     Model      : {args.model}")
        print(f"     Confidence : {res['confidence']*100:.1f}%")
        print(f"     Spam Prob  : {res['spam_prob']*100:.1f}%")
        print(f"     Ham Prob   : {res['ham_prob']*100:.1f}%")

    print("="*55 + "\n")


if __name__ == '__main__':
    _cli()
