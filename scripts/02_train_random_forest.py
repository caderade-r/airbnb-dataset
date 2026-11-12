"""Engineer 2: train one Random Forest to rank 12 destination categories."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from common import (read_users, make_features, make_preprocessor,
                    ranked_labels, CLASSES, TARGET, SEED, MODELS,
                    TABLES, ensure_dirs)


def main():
    ensure_dirs()
    df = read_users()
    X = make_features(df)
    y = df[TARGET].astype(str)
    ids = df['id'].astype(str)

    # Keep 20% of labeled users separate for an unbiased final comparison.
    row_ids = np.arange(len(df))
    train_rows, valid_rows = train_test_split(
        row_ids, test_size=0.20, random_state=SEED, stratify=y
    )
    pd.DataFrame({'train_row_index': train_rows}).to_csv(
        TABLES / 'train_row_indices.csv', index=False)
    pd.DataFrame({'validation_row_index': valid_rows}).to_csv(
        TABLES / 'validation_row_indices.csv', index=False)

    # The preprocessing steps are fitted only on the training users.
    model = Pipeline([
        ('preprocess', make_preprocessor()),
        ('forest', RandomForestClassifier(
            n_estimators=65,
            max_depth=17,
            min_samples_leaf=6,
            max_features='sqrt',
            random_state=SEED,
            n_jobs=2,
        )),
    ])
    print(f'Training Random Forest on {len(train_rows):,} users...')
    model.fit(X.iloc[train_rows], y.iloc[train_rows])
    print('Training complete.')

    # Save label probabilities, so evaluation can be run separately.
    probabilities = model.predict_proba(X.iloc[valid_rows])
    classes = model.named_steps['forest'].classes_
    top5 = ranked_labels(probabilities, classes, k=5)
    pred = classes[np.argmax(probabilities, axis=1)]
    out = pd.DataFrame({
        'id': ids.iloc[valid_rows].to_numpy(),
        'actual_destination': y.iloc[valid_rows].to_numpy(),
        'predicted_destination': pred,
    })
    for rank in range(5):
        out[f'top_{rank + 1}'] = top5[:, rank]
    for j, c in enumerate(classes):
        out[f'prob_{c}'] = probabilities[:, j]
    out.to_csv(TABLES / 'validation_predictions.csv', index=False)

    # Save the trained pipeline for later reuse on the competition test users.
    joblib.dump(model, MODELS / 'random_forest.joblib', compress=3)
    with open(MODELS / 'model_info.json', 'w') as f:
        json.dump({
            'classes': classes.tolist(), 'seed': SEED,
            'train_users': len(train_rows), 'validation_users': len(valid_rows),
            'model': 'RandomForestClassifier',
            'n_estimators': 65, 'max_depth': 17, 'min_samples_leaf': 6,
        }, f, indent=2)

    # Export feature importances after encoding text categories.
    preprocess = model.named_steps['preprocess']
    encoded_names = preprocess.get_feature_names_out()
    importances = model.named_steps['forest'].feature_importances_
    feature_table = pd.DataFrame({
        'feature': encoded_names, 'importance': importances
    }).sort_values('importance', ascending=False)
    feature_table.to_csv(TABLES / 'feature_importances.csv', index=False)

    # Output Kaggle-style top-five rows for the hidden-label competition test set.
    try:
        competition_test = read_users('test_users.csv')
        test_probs = model.predict_proba(make_features(competition_test))
        test_top5 = ranked_labels(test_probs, classes, k=5)
        submission = pd.DataFrame({
            'id': np.repeat(competition_test['id'].astype(str).to_numpy(), 5),
            'country': test_top5.reshape(-1),
        })
        submission.to_csv(TABLES / 'kaggle_submission.csv', index=False)
        print('Kaggle-style predictions saved (competition itself is historical).')
    except FileNotFoundError:
        print('No unlabeled competition test file; skipping submission export.')

    print(f'Validation predictions saved for {len(valid_rows):,} users.')


if __name__ == '__main__':
    main()
