# COMP472 Project : Visual Object Recognition and Sentiment Analysis
# Project Overview
Implementation and comparison of k-NN, Decision Tree, Random Forest and Neural Network on two
tasks built from pre-extracted feature vectors:

| Task | Features | Classes | Train | Hidden test |
|---|---|---|---|---|
| Image (object recognition) | 500 | 50 | 20,000 | 5,000 |
| Sentiment (TweetEval, BERT embeddings) | 768 | 3 | 12,000 | 5,000 |

A k-means clustering experiment was also conducted.


# Team Members 
Team : COMP472_Group09_F
| Name | Student ID | GitHub Username |
|---|---|---|
| Adja Boye  | xxx | [boyeadja](https://github.com/boyeadja) |
| Ange Akaliza | 402870048 |[Ellakaliza](https://github.com/Ellakaliza)  |
| Bethany Abebe | 40282242 | [BethanyABE](https://github.com/BethanyABE) |

## Setup

Run the following commands from the repository root to create and activate a virtual environment and install the necessary dependancies:

###Linux or macOS

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

###Windows Powershell

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
- Python: 3.12.3
- Key libraries: numpy, pandas, matplotlib scikit-learn, torch

Download the competition files from the Data tab of the [Kaggle competition](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia) in `data/` :
- `image_train.csv`
- `image_test.csv`
- `sentiment_train.csv`
- `sentiment_test.csv`
- `sample_submission.csv`.



## Repository layout

```
data/            competition CSVs (not committed)
splits/          saved train/validation indices (shared by all models)
src/             algorithm implementations, data and preprocessing code, nn_config.py
experiments/     one folder per algorithm: tuning, final predictions, analysis
results/         tables, figures and prediction files
reports/         report material and figures
submissions/     the file uploaded to Kaggle
```

## Reproducibility

- Random seed: **42** everywhere (Python, NumPy, PyTorch).
- Validation split: 80% train / 20% validation, stratified, created once by
  `get_split` in `preprocessing.py` and saved in `splits/<task>_split.npz`.
  **Every model uses these same files**, so validation accuracies and error
  comparisons are directly comparable. Do not regenerate or edit them.
- Preprocessing that learns parameters (standardization, PCA) is fitted on the
  training part only.

## How to run

### k-NN
```
python experiments/knn/run_knn.py            # sweep, plots, best settings
python experiments/knn/run_knn.py --final    # test predictions with the best settings
```

### Decision Tree and Random Forest

```
# Decision Tree: validation sweeps
python experiments/trees/run_trees.py --model tree --task image
python experiments/trees/run_trees.py --model tree --task sentiment

# Random Forest: validation sweeps
python experiments/trees/run_trees.py --model forest --task image
python experiments/trees/run_trees.py --model forest --task sentiment

# Generate final test predictions
python experiments/trees/run_trees.py --model tree --final
python experiments/trees/run_trees.py --model forest --final
```

### Neural network
Settings live in `src/nn_config.py` (`DEFAULTS` for tuning, `FINAL` for the chosen config).
```
python experiments/neural_network/tune.py            # one-setting-at-a-time tuning (edit TASK and RUN at the top)
python experiments/neural_network/nn_predictions.py  # validation accuracy + test predictions
python experiments/neural_network/error_analysis.py  # per-class metrics, confusion matrix, training curves
```
Final configs (chosen on the validation set):

| | Image | Sentiment |
|---|---|---|
| Hidden layers | (512, 256) | (256, 64) |
| Activation | ReLU | ReLU |
| Dropout | 0 | 0 |
| Learning rate | 1e-3 | 1e-4 |
| Weight decay | 1e-3 | 0 |
| Batch size | 64 | 64 |
| Optimizer | Adam | Adam |
| Epochs for final retrain | 44 | 5 |


### Clustering (k-means)
```
python experiments/kmeans/run_kmeans.py     # set TASK = "image" or "sentiment" at the top
```
