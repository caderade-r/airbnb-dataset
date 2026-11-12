"""Engineer 3: evaluate top-five destination ranking and classification."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from common import TABLES, FIGURES, CLASSES, ensure_dirs


def ndcg_at_5(actual, top_five):
    """Average gain from placing the correct country among the first five."""
    scores = []
    for truth, guesses in zip(actual, top_five):
        matches = np.flatnonzero(guesses == truth)
        if len(matches):
            rank = int(matches[0]) + 1
            scores.append(1.0 / np.log2(rank + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


def main():
    ensure_dirs()
    table_path = TABLES / 'validation_predictions.csv'
    if not table_path.exists():
        raise FileNotFoundError('Run scripts/02_train_random_forest.py first.')
    predictions = pd.read_csv(table_path, dtype={'id': str})
    actual = predictions['actual_destination'].to_numpy()
    predicted = predictions['predicted_destination'].to_numpy()
    top5 = predictions[[f'top_{i}' for i in range(1, 6)]].to_numpy()

    # Use overall training class frequencies as a simple ranking baseline.
    from common import read_users, TARGET
    rows = pd.read_csv(TABLES / 'train_row_indices.csv')['train_row_index'].to_numpy()
    train_labels = read_users().iloc[rows][TARGET]
    common5 = train_labels.value_counts().head(5).index.to_numpy()
    baseline_top5 = np.tile(common5, (len(actual), 1))
    baseline_top1 = np.full(len(actual), common5[0])

    metrics = {
        'baseline': {
            'ndcg_at_5': ndcg_at_5(actual, baseline_top5),
            'top_1_accuracy': accuracy_score(actual, baseline_top1),
            'macro_f1': f1_score(actual, baseline_top1, labels=CLASSES,
                                 average='macro', zero_division=0),
            'top_5_hit_rate': np.mean(np.any(baseline_top5 == actual[:, None], axis=1)),
        },
        'random_forest': {
            'ndcg_at_5': ndcg_at_5(actual, top5),
            'top_1_accuracy': accuracy_score(actual, predicted),
            'macro_f1': f1_score(actual, predicted, labels=CLASSES,
                                 average='macro', zero_division=0),
            'top_5_hit_rate': np.mean(np.any(top5 == actual[:, None], axis=1)),
        },
    }
    with open(TABLES / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(metrics).T.to_csv(TABLES / 'model_comparison.csv')
    print(pd.DataFrame(metrics).T.round(4).to_string())

    # Include all 12 classes, even if rare classes receive few predictions.
    report = pd.DataFrame(classification_report(
        actual, predicted, labels=CLASSES, target_names=CLASSES,
        output_dict=True, zero_division=0)).T
    report.to_csv(TABLES / 'per_class_metrics.csv')

    counts = confusion_matrix(actual, predicted, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(counts, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(title='Confusion matrix: Random Forest (validation)',
           xlabel='Predicted destination', ylabel='Actual destination')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Users')
    fig.tight_layout()
    fig.savefig(FIGURES / '06_confusion_matrix.png', dpi=180)
    plt.close(fig)

    imps = pd.read_csv(TABLES / 'feature_importances.csv').head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imps['feature'].str.replace('categories__', '', regex=False)
            .str.replace('numbers__', '', regex=False), imps['importance'])
    ax.set(title='Top 16 encoded features in Random Forest', xlabel='Feature importance')
    fig.tight_layout()
    fig.savefig(FIGURES / '07_feature_importance.png', dpi=180)
    plt.close(fig)

    comparisons = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, comparisons['ndcg_at_5'], width, label='NDCG@5')
    ax.bar(x + width / 2, comparisons['top_1_accuracy'], width, label='Top-1 accuracy')
    ax.set_xticks(x, ['Frequency baseline', 'Random Forest'])
    ax.set(ylim=(0, 1), ylabel='Score', title='Model performance on held-out users')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / '08_model_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
