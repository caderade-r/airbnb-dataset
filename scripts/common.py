"""Shared dataset settings and preprocessing for the Airbnb project."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
TABLES = RESULTS / 'tables'
MODELS = ROOT / 'models'
SEED = 42
TARGET = 'country_destination'
CLASSES = ['US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU', 'NDF', 'other']

CATEGORICAL = [
    'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
    'signup_app', 'first_device_type', 'first_browser',
]
NUMERIC = [
    'age', 'account_year', 'account_month', 'account_weekday',
    'first_active_year', 'first_active_month', 'first_active_weekday',
]
FEATURES = NUMERIC + CATEGORICAL


def ensure_dirs():
    for folder in [DATA, FIGURES, TABLES, MODELS]:
        folder.mkdir(parents=True, exist_ok=True)


def read_users(filename='train_users_2.csv'):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Extract this file from the Airbnb dataset archive into data/.'
        )
    return pd.read_csv(path)


def make_features(df):
    """Make one model row per user from information available around signup."""
    X = df.copy()

    # Remove impossible age values before calculating the training median.
    X['age'] = pd.to_numeric(X['age'], errors='coerce')
    X.loc[~X['age'].between(18, 100), 'age'] = np.nan

    # Convert dates into year, month, and weekday values.
    created = pd.to_datetime(X['date_account_created'], errors='coerce')
    active = pd.to_datetime(X['timestamp_first_active'].astype('string'),
                            format='%Y%m%d%H%M%S', errors='coerce')
    X['account_year'] = created.dt.year
    X['account_month'] = created.dt.month
    X['account_weekday'] = created.dt.dayofweek
    X['first_active_year'] = active.dt.year
    X['first_active_month'] = active.dt.month
    X['first_active_weekday'] = active.dt.dayofweek

    # Sign-up flow is a category code, not a continuous measurement.
    X['signup_flow'] = X['signup_flow'].astype('string')

    # Preserve missing categorical values as a separate group.
    for column in CATEGORICAL:
        X[column] = X[column].astype('string').replace('-unknown-', pd.NA)
        X[column] = X[column].fillna('Missing').astype(str)

    # Keep only columns used to predict the destination.
    return X[FEATURES].copy()


def make_preprocessor():
    """Fit missing-value handling and encoding on training rows only."""
    numeric_steps = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
    ])
    categorical_steps = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('numbers', numeric_steps, NUMERIC),
        ('categories', categorical_steps, CATEGORICAL),
    ])


def ranked_labels(probabilities, labels, k=5):
    """Return the k labels with highest probability for each row."""
    order = np.argsort(-probabilities, axis=1)[:, :k]
    labels = np.asarray(labels)
    return labels[order]
