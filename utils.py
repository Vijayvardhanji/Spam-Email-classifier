"""
utils.py
--------
Shared utility functions for the Spam Email Classifier project.
Includes: metric computation, plotting helpers, model persistence.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    'spam':      '#E74C3C',   # red
    'ham':       '#2ECC71',   # green
    'primary':   '#2C3E50',   # dark navy
    'secondary': '#3498DB',   # blue
    'accent':    '#F39C12',   # orange
    'bg':        '#F8F9FA',   # light grey
}


# ── Metric helpers ───────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute classification metrics and print a formatted report.

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    metrics = {
        'model':     model_name,
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} – Performance Report")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))

    return metrics


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_class_distribution(df, save_path: str = None):
    """Bar chart showing Ham vs Spam distribution."""
    counts = df['label'].value_counts()
    colors = [PALETTE['ham'], PALETTE['spam']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white',
                       linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30,
                     f'{val:,}\n({val/len(df)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold',
                      color=PALETTE['primary'], pad=15)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_facecolor(PALETTE['bg'])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=counts.index,
        colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    axes[1].set_title('Spam vs Ham Ratio', fontsize=14, fontweight='bold',
                      color=PALETTE['primary'], pad=15)

    plt.suptitle('Dataset Distribution', fontsize=16, fontweight='bold',
                 color=PALETTE['primary'], y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=PALETTE['bg'])
        print(f"[INFO] Saved class distribution → {save_path}")
    return fig


def plot_word_frequency(word_freq_ham: dict, word_freq_spam: dict,
                        top_n: int = 15, save_path: str = None):
    """Horizontal bar charts for top words in Ham and Spam."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE['bg'])

    for ax, (freq, label, color) in zip(
        axes,
        [(word_freq_ham, 'Ham', PALETTE['ham']),
         (word_freq_spam, 'Spam', PALETTE['spam'])]
    ):
        words  = list(freq.keys())[:top_n]
        values = list(freq.values())[:top_n]
        y_pos  = np.arange(len(words))

        bars = ax.barh(y_pos, values, color=color, alpha=0.85,
                       edgecolor='white', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_title(f'Top {top_n} Words – {label}', fontsize=13,
                     fontweight='bold', color=PALETTE['primary'], pad=12)
        ax.set_facecolor(PALETTE['bg'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontsize=8)

    plt.suptitle('Word Frequency Analysis', fontsize=16, fontweight='bold',
                 color=PALETTE['primary'])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=PALETTE['bg'])
        print(f"[INFO] Saved word frequency chart → {save_path}")
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name: str,
                          save_path: str = None):
    """Styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Ham', 'Spam']

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE['bg'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    ax.set_title(f'Confusion Matrix – {model_name}',
                 fontsize=14, fontweight='bold',
                 color=PALETTE['primary'], pad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=PALETTE['bg'])
        print(f"[INFO] Saved confusion matrix → {save_path}")
    return fig


def plot_accuracy_comparison(metrics_list: list, save_path: str = None):
    """
    Grouped bar chart comparing Naive Bayes vs SVM across all metrics.

    Args:
        metrics_list (list): List of dicts returned by compute_metrics().
    """
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metric_labels))
    width = 0.3
    colors = [PALETTE['secondary'], PALETTE['accent']]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(PALETTE['bg'])

    for i, (metrics, color) in enumerate(zip(metrics_list, colors)):
        values = [metrics[m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width,
                      label=metrics['model'], color=color,
                      edgecolor='white', linewidth=1.2, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.set_title('Model Performance Comparison: Naive Bayes vs SVM',
                 fontsize=14, fontweight='bold', color=PALETTE['primary'], pad=15)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.set_facecolor(PALETTE['bg'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=PALETTE['bg'])
        print(f"[INFO] Saved comparison chart → {save_path}")
    return fig


# ── Model persistence ────────────────────────────────────────────────────────

def save_model(obj, path: str):
    """Serialize a Python object to disk with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"[INFO] Saved → {path}")


def load_model(path: str):
    """Load a joblib-serialised object from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    print(f"[INFO] Loaded ← {path}")
    return obj
