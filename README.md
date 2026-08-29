 **Comparative Analysis of Machine Learning Models for Dry Bean Classification**

**Project Overview:**

This project presents a comparative analysis of machine learning classification algorithms for identifying seven different varieties of dry beans based on their morphological characteristics.

The main objective is to compare the performance of multiple classification algorithms and evaluate how consistently they perform across different random data splits.

Ten machine learning models were evaluated using three different random states. The models were compared using accuracy, precision, recall, F1-score, and ROC-AUC.

The Multi-layer Perceptron achieved the best overall performance based on the mean F1-score across the three random states.

**Dataset**

The Dry Bean dataset contains 13,611 observations and 17 columns.

The dataset contains 16 numerical morphological features and one categorical target variable called Class.

The seven dry bean varieties are:

DERMASON

SIRA

SEKER

HOROZ

CALI

BARBUNYA

BOMBAY

The original dataset contained 68 duplicate observations. These duplicate observations were removed before model training.

**Features:**

The following 16 morphological features were used as input variables:

Area

Perimeter

MajorAxisLength

MinorAxisLength

AspectRation

Eccentricity

ConvexArea

EquivDiameter

Extent

Solidity

roundness

Compactness

ShapeFactor1

ShapeFactor2

ShapeFactor3

ShapeFactor4

The target variable is Class.

**Data Preprocessing:**

The following preprocessing steps were applied:

1. The dataset was loaded using Pandas.
2. Duplicate observations were removed.
3. The Class variable was encoded into numerical labels.
4. Stratified sampling was used to preserve the class distribution.
5. The dataset was divided into 80 percent training data, 10 percent validation data, and 10 percent testing data.
6. StandardScaler was applied to models that require feature scaling.
7. The experiments were repeated using three random states: 42, 100, and 2026.

The validation set was used for model selection, while the test set was kept for final performance evaluation.

**Machine Learning Models:**

The following ten classification algorithms were compared:

1. Logistic Regression
2. Linear Discriminant Analysis
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. K-Nearest Neighbors
7. Gaussian Naive Bayes
8. AdaBoost
9. Support Vector Machine
10. Multi-layer Perceptron

**Experimental Design:**

Each model was evaluated using three different random states:

42

100

2026

Using multiple random states helps determine whether model performance remains consistent when the training, validation, and testing data are divided differently.

For each random state, the models were trained using the training data and evaluated on the validation data.

The model with the highest validation F1-score was selected for that random state.

The selected model was then trained using the combined training and validation data and evaluated on the previously unseen test data.

**Evaluation Metrics:**

The models were evaluated using the following metrics:

Accuracy

Accuracy measures the proportion of correctly classified observations.

Precision

Precision measures how many of the observations predicted as a particular class were actually members of that class.

Recall

Recall measures how many observations belonging to a particular class were correctly identified.

F1-score

F1-score provides a balance between precision and recall. It was used as the primary metric for model comparison.

ROC-AUC

ROC-AUC measures the ability of the model to distinguish between different classes.

**Results:**

The Multi-layer Perceptron achieved the best overall performance across the three random states.

Its average performance was:

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.9292 | 0.0092 |
| Precision | 0.9291 | Not calculated |
| Recall | 0.9292 | Not calculated |
| F1-score | 0.9289 | 0.0092 |
| ROC-AUC | 0.9947 | 0.0012 |

The Multi-layer Perceptron achieved a mean F1-score of 0.9289 across the three random states.

Its F1-score standard deviation was 0.0092, indicating relatively stable performance across the different data splits.

The mean ROC-AUC was 0.9947, indicating strong class discrimination.

**Best Model by Random State:**

The best model based on validation F1-score was:

| Random State | Best Model | Validation F1-score |
|---:|---|---:|
| 42 | Multi-layer Perceptron | 0.9217 |
| 100 | Multi-layer Perceptron | 0.9393 |
| 2026 | Support Vector Machine | 0.9265 |

The Multi-layer Perceptron was selected for two of the three random states, while the Support Vector Machine was selected for one random state.

**Per-Class Analysis:**

Per-class performance was also evaluated to examine how well the models classified individual dry bean varieties.

The per-class results are available in:

results/metrics/per_class_summary.csv

The corresponding visualization is available in:

figures/per_class_f1.png

This analysis is useful because the dataset contains an unequal number of observations across the seven classes.

**Visualizations:**

The project contains visualizations showing:

Model F1-score comparison

Accuracy across random states

Precision across random states

Recall across random states

F1-score across random states

ROC-AUC across random states

Per-class F1-score

Confusion matrices

**Project Structure:**

Dry Bean Dataset

README.md

data

Dry_Bean_Dataset.csv

figures

all_models_f1_comparison.png

accuracy_by_seed.png

precision_by_seed.png

recall_by_seed.png

f1_by_seed.png

roc_auc_by_seed.png

per_class_f1.png

results

metrics

all_seed_results.csv

final_model_results.csv

overall_model_comparison.csv

per_class_summary.csv

report_results.csv

models

src

eda.py

preprocessing.py

models.py

train.py

visualization.py

final_experiment.py

per_class_analysis.py

overall_model_comparison.py

best_model.py

report_results.py

**Technologies Used:**

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Visual Studio Code

**Key Findings:**

The comparison showed that the Multi-layer Perceptron achieved the strongest overall performance based on the mean F1-score across the three random states.

The Multi-layer Perceptron achieved a mean accuracy of 0.9292 and a mean F1-score of 0.9289.

The model also achieved a mean ROC-AUC of 0.9947.

The Support Vector Machine also demonstrated strong performance and was selected as the best model for the random state 2026.

The results indicate that morphological characteristics can be used effectively to distinguish between different dry bean varieties using machine learning classification algorithms.

**Limitations:**

The experiments were conducted using three random states.

Only the selected machine learning algorithms were evaluated.

Extensive hyperparameter optimization was not performed.

The analysis is based on the available morphological features in the dataset.

**Future Work:**


Additional cross-validation experiments could be performed to obtain more robust estimates of model performance.

Feature selection techniques could be investigated to determine which morphological features contribute most to classification.

Ensemble methods could be further explored.

The machine learning models could also be compared with deep learning approaches.

**Conclusion:**

This project demonstrated a systematic comparison of ten machine learning classification algorithms for dry bean variety classification.

The models were evaluated using three random states and multiple performance metrics.

Among the evaluated models, the Multi-layer Perceptron achieved the best overall mean F1-score of 0.9289 and a mean ROC-AUC of 0.9947.

The results demonstrate that machine learning can provide strong performance for classifying dry bean varieties using morphological characteristics.
