# airbnb-dataset
Data mining project for predicting where new users will book their first stay using using AirBnB.
Contains the code, data, trained model, and results for the core Random Forest classification project. It does not include the draft report or presentation.

# Run from the project folder

1. Install Python and run `python -m pip install -r requirements.txt` in the terminal.
2. Run the entire project with `python run_all.py`.
3. The scripts run in order: EDA, Random Forest training, then evaluation.
4. Look in `results/figures/` for charts and `results/tables/` for the output CSVs.

# Main files

- `scripts/01_eda.py` — explores the labeled data, saves graphs and data quality summaries.
- `scripts/02_train_random_forest.py` — preprocesses features, splits the data 80/20, trains the Random Forest, saves probabilities, predictions, and the model.
- `scripts/03_evaluate.py` — computes NDCG@5, accuracy, macro F1 and a frequency baseline, then saves evaluation charts.
- `scripts/common.py` — shared paths, feature list, cleaning rules, and encoding functions. Required by all 3 scripts.
- `data/train_users_2.csv` — labeled users; `data/test_users.csv` — unlabeled competition users.
- `requirements.txt` — Python packages.
- `run_all.py` — executes 01, 02, and 03 in the correct order.

The current results are saved from an earlier run. Rerun the scripts to reproduce them. Training can take time.

The official Kaggle competition used a chronological test split; this project uses stratified 80/20 held-out validation for classroom analysis. No session-log features are included in the current model.
