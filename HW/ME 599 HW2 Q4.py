import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

# ------------------------------------------------------------
# Step 1: Load the dataset
# ------------------------------------------------------------

df = pd.read_csv("ai4i2020.csv")

print("Dataset shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Step 2: Define target and remove leakage columns
# ------------------------------------------------------------

target_col = "Machine failure"

# These columns directly describe specific failure modes.
# Do not use them as predictors for binary machine failure.
failure_mode_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# ID columns are not useful predictive features.
id_cols = ["UDI", "Product ID"]

drop_cols = [target_col]

for col in failure_mode_cols + id_cols:
    if col in df.columns:
        drop_cols.append(col)

X = df.drop(columns=drop_cols)
y = df[target_col].astype(int)

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())
print("\nTarget distribution percentage:")
print(y.value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# Step 3: Train / validation / test split using 3:1:1 ratio
# ------------------------------------------------------------
# 3:1:1 means:
# training = 60%
# validation = 20%
# test = 20%

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation target distribution:")
print(y_val.value_counts(normalize=True) * 100)

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# Step 4: Identify numerical and categorical columns
# ------------------------------------------------------------

categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

print("\nNumerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)

# ------------------------------------------------------------
# Step 5: Helper function for OneHotEncoder compatibility
# ------------------------------------------------------------

def make_one_hot_encoder():
    """
    Handles both older and newer versions of scikit-learn.
    Newer versions use sparse_output.
    Older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ------------------------------------------------------------
# Step 6: Build preprocessors
# ------------------------------------------------------------

def make_preprocessor(scale_numeric):
    """
    Logistic regression needs scaled numerical features.
    Random forest does not require scaling.
    """

    if scale_numeric:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
    else:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor

# ------------------------------------------------------------
# Step 7: Define model-building functions
# ------------------------------------------------------------

def make_logistic_regression(C_value):
    """
    Logistic regression model.
    class_weight='balanced' helps with imbalanced failure data.
    """

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(
                C=C_value,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            ))
        ]
    )

    return model


def make_random_forest(n_estimators_value, max_depth_value, min_samples_leaf_value):
    """
    Random forest model.
    class_weight='balanced_subsample' helps with imbalanced failure data.
    """

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators_value,
                max_depth=max_depth_value,
                min_samples_leaf=min_samples_leaf_value,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    return model

# ------------------------------------------------------------
# Step 8: Use the validation set to choose simple hyperparameters
# ------------------------------------------------------------
# The test set is not used for model selection.
# Choose the best model within each family based on validation macro F1.

logistic_candidates = {
    "Logistic Regression, C=0.1": make_logistic_regression(C_value=0.1),
    "Logistic Regression, C=1.0": make_logistic_regression(C_value=1.0),
    "Logistic Regression, C=10.0": make_logistic_regression(C_value=10.0)
}

random_forest_candidates = {
    "Random Forest, depth=None, leaf=1": make_random_forest(
        n_estimators_value=500,
        max_depth_value=None,
        min_samples_leaf_value=1
    ),
    "Random Forest, depth=10, leaf=2": make_random_forest(
        n_estimators_value=500,
        max_depth_value=10,
        min_samples_leaf_value=2
    ),
    "Random Forest, depth=15, leaf=2": make_random_forest(
        n_estimators_value=500,
        max_depth_value=15,
        min_samples_leaf_value=2
    )
}


def select_best_model(candidate_models, family_name):
    """
    Fits each candidate on the training set and evaluates on validation set.
    Selects the model with the best validation macro F1.
    """

    validation_results = []

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
        failure_f1 = f1_score(y_val, y_val_pred, pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_val, y_val_pred)

        validation_results.append({
            "family": family_name,
            "model_name": model_name,
            "model": model,
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
            "validation_failure_f1": failure_f1,
            "validation_accuracy": accuracy
        })

    validation_results_df = pd.DataFrame(validation_results)

    print("\nValidation results for", family_name)
    print(validation_results_df.drop(columns=["model"]))

    best_index = validation_results_df["validation_macro_f1"].idxmax()
    best_row = validation_results_df.loc[best_index]

    print("\nBest", family_name, "model:")
    print(best_row["model_name"])

    return best_row["model"], best_row["model_name"], validation_results_df.drop(columns=["model"])


best_logistic_model, best_logistic_name, logistic_validation_results = select_best_model(
    logistic_candidates,
    "Logistic Regression"
)

best_rf_model, best_rf_name, rf_validation_results = select_best_model(
    random_forest_candidates,
    "Random Forest"
)

# ------------------------------------------------------------
# Step 9: Evaluate final selected models on the test set
# ------------------------------------------------------------

def evaluate_on_test_set(model, model_name):
    """
    Prints confusion matrix and classification report on the test set.
    """

    y_test_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Actual non-failure (0)", "Actual failure (1)"],
        columns=["Predicted non-failure (0)", "Predicted failure (1)"]
    )

    print("\n====================================================")
    print("Test results for:", model_name)
    print("====================================================")

    print("\nTest confusion matrix:")
    print(cm_df)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["Non-failure class (0)", "Failure class (1)"],
            digits=4,
            zero_division=0
        )
    )

    report_dict = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["Non-failure class (0)", "Failure class (1)"],
        digits=4,
        output_dict=True,
        zero_division=0
    )

    return cm_df, report_dict, y_test_pred


logistic_cm, logistic_report, logistic_test_pred = evaluate_on_test_set(
    best_logistic_model,
    best_logistic_name
)

rf_cm, rf_report, rf_test_pred = evaluate_on_test_set(
    best_rf_model,
    best_rf_name
)

# ------------------------------------------------------------
# Step 10: Create a comparison table for Question 4(b)
# ------------------------------------------------------------

def extract_metrics(report_dict, model_name):
    """
    Extract class-wise, macro, weighted, and accuracy values.
    """

    rows = []

    for label in [
        "Non-failure class (0)",
        "Failure class (1)",
        "macro avg",
        "weighted avg"
    ]:
        rows.append({
            "model": model_name,
            "metric_level": label,
            "precision": report_dict[label]["precision"],
            "recall": report_dict[label]["recall"],
            "f1_score": report_dict[label]["f1-score"],
            "support": report_dict[label]["support"]
        })

    rows.append({
        "model": model_name,
        "metric_level": "accuracy",
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": report_dict["accuracy"],
        "support": report_dict["macro avg"]["support"]
    })

    return rows


comparison_rows = []
comparison_rows.extend(extract_metrics(logistic_report, best_logistic_name))
comparison_rows.extend(extract_metrics(rf_report, best_rf_name))

comparison_df = pd.DataFrame(comparison_rows)

print("\n====================================================")
print("Final comparison table")
print("====================================================")
print(comparison_df)

# Save results to CSV
comparison_df.to_csv("model_comparison_results.csv", index=False)

print("\nSaved comparison table to model_comparison_results.csv")