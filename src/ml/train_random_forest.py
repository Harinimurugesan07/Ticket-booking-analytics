import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)



# CONFIGURATION


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

ML_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ml"
)

ENCODED_DIR = os.path.join(
    ML_DIR,
    "encoded"
)

MODEL_DIR = os.path.join(
    ML_DIR,
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "ml"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)



# LOAD DATA


def load_data():

    X_train = pd.read_csv(
        os.path.join(
            ENCODED_DIR,
            "X_train.csv"
        )
    )

    X_test = pd.read_csv(
        os.path.join(
            ENCODED_DIR,
            "X_test.csv"
        )
    )

    y_train = pd.read_csv(
        os.path.join(
            ENCODED_DIR,
            "y_train.csv"
        )
    )["is_cancelled"]

    y_test = pd.read_csv(
        os.path.join(
            ENCODED_DIR,
            "y_test.csv"
        )
    )["is_cancelled"]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



# TRAIN RANDOM FOREST


def train_model(
    X_train,
    y_train
):

    print("\n")
    
    print("TRAINING RANDOM FOREST")
    

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_split=5,

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nRandom Forest training completed.")

    return model



# EVALUATE MODEL


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n")
    
    print("RANDOM FOREST PERFORMANCE")
    

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "predictions": predictions,
        "probabilities": probabilities
    }



# FEATURE IMPORTANCE


def show_feature_importance(
    model,
    X_train
):

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n")
    
    print("TOP 15 FEATURE IMPORTANCES")
    

    print(
        importance
        .head(15)
        .to_string(index=False)
    )

    return importance



# SAVE MODEL


def save_model(model):

    model_path = os.path.join(
        MODEL_DIR,
        "random_forest.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved:")
    print(model_path)

    return model_path



# SAVE METRICS


def save_metrics(metrics):

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"]
            }
        ]
    )

    metrics_path = os.path.join(
        REPORT_DIR,
        "random_forest_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("\nMetrics saved:")
    print(metrics_path)



# SAVE PREDICTIONS


def save_predictions(
    y_test,
    predictions,
    probabilities
):

    prediction_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": predictions,
            "cancellation_probability": probabilities
        }
    )

    prediction_path = os.path.join(
        REPORT_DIR,
        "random_forest_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print("\nPredictions saved:")
    print(prediction_path)



# SAVE FEATURE IMPORTANCE


def save_feature_importance(
    importance
):

    importance_path = os.path.join(
        REPORT_DIR,
        "random_forest_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    print("\nFeature importance saved:")
    print(importance_path)



# MAIN


def main():

    
    print("STEP 10.4 — RANDOM FOREST CLASSIFIER")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    print(
        f"\nX_train shape : {X_train.shape}"
    )

    print(
        f"X_test shape  : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = show_feature_importance(
        model,
        X_train
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        model
    )

    save_metrics(
        metrics
    )

    save_predictions(
        y_test,
        metrics["predictions"],
        metrics["probabilities"]
    )

    save_feature_importance(
        importance
    )

    print("\n")
    
    print("STEP 10.4 COMPLETED")
    


if __name__ == "__main__":
    main()