import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
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

FIGURE_DIR = os.path.join(
    REPORT_DIR,
    "figures"
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)



# LOAD DATA


def load_data():

    X_test = pd.read_csv(
        os.path.join(
            ENCODED_DIR,
            "X_test.csv"
        )
    )

    y_test = pd.read_csv(
        os.path.join(
            ENCODED_DIR,
            "y_test.csv"
        )
    )["is_cancelled"]

    return X_test, y_test



# LOAD MODELS


def load_models():

    logistic_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "logistic_regression.pkl"
        )
    )

    random_forest_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "random_forest.pkl"
        )
    )

    return (
        logistic_model,
        random_forest_model
    )



# CONFUSION MATRIX


def create_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name
):

    predictions = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n")
    
    print(f"{model_name.upper()} — CONFUSION MATRIX")
    

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Cancelled",
            "Cancelled"
        ]
    )

    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    filepath = os.path.join(
        FIGURE_DIR,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved: {filepath}"
    )



# ROC CURVE


def create_roc_curve(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
):

    logistic_probability = (
        logistic_model
        .predict_proba(X_test)[:, 1]
    )

    random_forest_probability = (
        random_forest_model
        .predict_proba(X_test)[:, 1]
    )

    logistic_fpr, logistic_tpr, _ = roc_curve(
        y_test,
        logistic_probability
    )

    random_forest_fpr, random_forest_tpr, _ = roc_curve(
        y_test,
        random_forest_probability
    )

    logistic_auc = roc_auc_score(
        y_test,
        logistic_probability
    )

    random_forest_auc = roc_auc_score(
        y_test,
        random_forest_probability
    )

    print("\n")
    
    print("ROC-AUC COMPARISON")
    

    print(
        f"\nLogistic Regression ROC-AUC : "
        f"{logistic_auc:.4f}"
    )

    print(
        f"Random Forest ROC-AUC       : "
        f"{random_forest_auc:.4f}"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label=f"Logistic Regression (AUC={logistic_auc:.3f})"
    )

    plt.plot(
        random_forest_fpr,
        random_forest_tpr,
        label=f"Random Forest (AUC={random_forest_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve — Cancellation Prediction"
    )

    plt.legend()

    plt.tight_layout()

    filepath = os.path.join(
        FIGURE_DIR,
        "roc_curve_comparison.png"
    )

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved: {filepath}"
    )



# PRECISION-RECALL CURVE


def create_precision_recall_curve(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
):

    logistic_probability = (
        logistic_model
        .predict_proba(X_test)[:, 1]
    )

    random_forest_probability = (
        random_forest_model
        .predict_proba(X_test)[:, 1]
    )

    logistic_precision, logistic_recall, _ = (
        precision_recall_curve(
            y_test,
            logistic_probability
        )
    )

    random_forest_precision, random_forest_recall, _ = (
        precision_recall_curve(
            y_test,
            random_forest_probability
        )
    )

    logistic_ap = average_precision_score(
        y_test,
        logistic_probability
    )

    random_forest_ap = average_precision_score(
        y_test,
        random_forest_probability
    )

    print("\n")
    
    print("PRECISION-RECALL ANALYSIS")
    

    print(
        f"\nLogistic Regression AP : "
        f"{logistic_ap:.4f}"
    )

    print(
        f"Random Forest AP       : "
        f"{random_forest_ap:.4f}"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        logistic_recall,
        logistic_precision,
        label=f"Logistic Regression (AP={logistic_ap:.3f})"
    )

    plt.plot(
        random_forest_recall,
        random_forest_precision,
        label=f"Random Forest (AP={random_forest_ap:.3f})"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve — Cancellation Prediction"
    )

    plt.legend()

    plt.tight_layout()

    filepath = os.path.join(
        FIGURE_DIR,
        "precision_recall_comparison.png"
    )

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved: {filepath}"
    )



# RANDOM FOREST FEATURE IMPORTANCE


def analyze_feature_importance(
    random_forest_model,
    X_test
):

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": (
                random_forest_model
                .feature_importances_
            )
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
    
    print("TOP 20 RANDOM FOREST FEATURES")
    

    print(
        importance
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Plot top 15
    # --------------------------------------------------------

    top_features = (
        importance
        .head(15)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Feature Importance — Random Forest"
    )

    plt.tight_layout()

    filepath = os.path.join(
        FIGURE_DIR,
        "random_forest_feature_importance.png"
    )

    plt.savefig(
        filepath,
        dpi=150
    )

    plt.close()

    print(
        f"\nSaved: {filepath}"
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        REPORT_DIR,
        "detailed_feature_importance.csv"
    )

    importance.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Saved: {csv_path}"
    )



# FINAL INTERPRETATION


def print_interpretation(
    logistic_model,
    random_forest_model,
    X_test,
    y_test
):

    logistic_probability = (
        logistic_model
        .predict_proba(X_test)[:, 1]
    )

    random_forest_probability = (
        random_forest_model
        .predict_proba(X_test)[:, 1]
    )

    logistic_auc = roc_auc_score(
        y_test,
        logistic_probability
    )

    random_forest_auc = roc_auc_score(
        y_test,
        random_forest_probability
    )

    print("\n")
    
    print("MODEL EVALUATION SUMMARY")
    

    if random_forest_auc > logistic_auc:

        print(
            "\nRandom Forest currently provides "
            "the stronger ROC-AUC performance."
        )

    elif logistic_auc > random_forest_auc:

        print(
            "\nLogistic Regression currently provides "
            "the stronger ROC-AUC performance."
        )

    else:

        print(
            "\nBoth models have the same ROC-AUC."
        )

    print(
        "\nThese models can now be evaluated further "
        "using business-specific cancellation-risk thresholds."
    )



# MAIN


def main():

    
    print("STEP 10.6 — DETAILED MODEL EVALUATION")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X_test, y_test = load_data()

    (
        logistic_model,
        random_forest_model
    ) = load_models()

    print(
        f"\nTest records: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    create_confusion_matrix(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    create_confusion_matrix(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    create_roc_curve(
        logistic_model,
        random_forest_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Precision Recall
    # --------------------------------------------------------

    create_precision_recall_curve(
        logistic_model,
        random_forest_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    analyze_feature_importance(
        random_forest_model,
        X_test
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print_interpretation(
        logistic_model,
        random_forest_model,
        X_test,
        y_test
    )

    print("\n")
    
    print("STEP 10.6 COMPLETED")
    


if __name__ == "__main__":
    main()