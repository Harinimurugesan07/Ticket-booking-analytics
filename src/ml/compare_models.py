import os
import pandas as pd



# CONFIGURATION


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "ml"
)



# LOAD MODEL METRICS


def load_metrics():

    logistic_path = os.path.join(
        REPORT_DIR,
        "logistic_regression_metrics.csv"
    )

    random_forest_path = os.path.join(
        REPORT_DIR,
        "random_forest_metrics.csv"
    )

    logistic = pd.read_csv(
        logistic_path
    )

    random_forest = pd.read_csv(
        random_forest_path
    )

    return logistic, random_forest



# COMPARE MODELS


def compare_models(
    logistic,
    random_forest
):

    comparison = pd.concat(
        [
            logistic,
            random_forest
        ],
        ignore_index=True
    )

    return comparison



# DETERMINE BEST MODEL


def determine_best_model(
    comparison
):

    # For cancellation prediction,
    # ROC-AUC is our primary comparison metric.

    best_index = comparison[
        "roc_auc"
    ].idxmax()

    best_model = comparison.loc[
        best_index
    ]

    return best_model



# PRINT COMPARISON


def print_comparison(
    comparison,
    best_model
):

    print("\n")
    
    print("MODEL COMPARISON")
    

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    print(
        comparison[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n")
    
    print("BEST MODEL")
    

    print(
        f"\nModel selected: "
        f"{best_model['model']}"
    )

    print(
        f"ROC-AUC: "
        f"{best_model['roc_auc']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{best_model['f1_score']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_model['recall']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_model['precision']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{best_model['accuracy']:.4f}"
    )



# BUSINESS INTERPRETATION


def business_interpretation(
    comparison,
    best_model
):

    print("\n")
    
    print("BUSINESS INTERPRETATION")
    

    print(
        "\nThe cancellation prediction models were "
        "evaluated using multiple classification metrics."
    )

    print(
        "\nROC-AUC was used as the primary model-selection "
        "metric because the business needs to distinguish "
        "between likely cancelled and non-cancelled bookings."
    )

    print(
        f"\nThe selected model is "
        f"{best_model['model']}."
    )

    print(
        f"It achieved a ROC-AUC of "
        f"{best_model['roc_auc']:.4f}."
    )

    print(
        "\nThe selected model can later be used to "
        "assign a cancellation-risk probability "
        "to new bookings."
    )



# SAVE COMPARISON


def save_comparison(
    comparison
):

    output_path = os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False
    )

    print("\nComparison saved:")
    print(output_path)



# MAIN


def main():

    
    print("STEP 10.5 — MODEL COMPARISON")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    logistic, random_forest = load_metrics()

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    comparison = compare_models(
        logistic,
        random_forest
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = determine_best_model(
        comparison
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_comparison(
        comparison,
        best_model
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    business_interpretation(
        comparison,
        best_model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_comparison(
        comparison
    )

    print("\n")
    
    print("STEP 10.5 COMPLETED")
    


if __name__ == "__main__":
    main()