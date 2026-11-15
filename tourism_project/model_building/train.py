# importing libraries
import os
import subprocess

if not os.path.exists("Xtrain.csv"):
    print("Running data preparation script...")
    subprocess.run(["python", "tourism_project/model_building/prep.py"], check=True)

# for data manipulation
import pandas as pd

# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb

# for model serialization and experiment tracking
import joblib
import mlflow
from pathlib import Path

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# extracting numerical variable
numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# extracting categorical variables
categorical_features = Xtrain.select_dtypes(
    include=["object"]
).columns.tolist()

# printing numerical and categorical variables
print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# Setting the class weight to handle class imbalance
class_weight = (
    ytrain.value_counts()[0]
    / ytrain.value_counts()[1]
)

print(f"Scale Pos Weight: {class_weight:.2f}")

# Defining the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# defining the base XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=class_weight,
    random_state=42
)

# Model pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# Defining hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__colsample_bytree": [0.4, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.6]
}

# Training the Model with MLflow Tracking
with mlflow.start_run():

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1"
    )

    grid_search.fit(Xtrain, ytrain)

    # Logging all parameter combinations
    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(results["params"][i])

            mlflow.log_metric(
                "mean_test_score",
                results["mean_test_score"][i]
            )

            mlflow.log_metric(
                "std_test_score",
                results["std_test_score"][i]
            )

    # Storing the best model
    best_model = grid_search.best_estimator_

    mlflow.log_params(
        grid_search.best_params_
    )

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    # Setting classification threshold
    classification_threshold = 0.55

    # Making predictions on the training data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    # Making predictions on the test data
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

  
    # Evaluating the reports
    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )

    # Logging the metrics
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"]
    })

   
    # Saving the Model for Streamlit Deployment
    project_root = Path("tourism_project")

    deployment_dir = project_root / "deployment"

    deployment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = deployment_dir / "best_predictor_model_v1.joblib"

    joblib.dump(
        best_model,
        model_path
    )

    mlflow.log_artifact(
        str(model_path),
        artifact_path="model"
    )

    print("\nModel saved successfully")

    print("Model Location:")
    print(model_path.resolve())

    print(
        "File Exists:",
        model_path.exists()
    )
