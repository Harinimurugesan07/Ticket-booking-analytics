import os
import joblib
import pandas as pd



# STEP 10.7
# CANCELLATION RISK SCORING




# PROJECT PATHS


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
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


# Make sure report directory exists
os.makedirs(
    REPORT_DIR,
    exist_ok=True
)



# LOAD BEST MODEL


def load_best_model():

    comparison_path = os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    )

    if not os.path.exists(comparison_path):

        raise FileNotFoundError(
            f"\nModel comparison file not found:\n"
            f"{comparison_path}\n\n"
            f"Run STEP 10.5 first."
        )

    comparison = pd.read_csv(
        comparison_path
    )

    if comparison.empty:

        raise ValueError(
            "model_comparison.csv is empty."
        )

    # Select model with highest ROC-AUC
    best_index = comparison[
        "roc_auc"
    ].idxmax()

    best_model_name = comparison.loc[
        best_index,
        "model"
    ]

    # --------------------------------------------------------
    # Determine model path
    # --------------------------------------------------------

    if best_model_name == "Random Forest":

        model_filename = (
            "random_forest.pkl"
        )

    elif best_model_name == "Logistic Regression":

        model_filename = (
            "logistic_regression.pkl"
        )

    else:

        raise ValueError(
            f"Unknown model found in comparison file: "
            f"{best_model_name}"
        )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nModel file not found:\n"
            f"{model_path}"
        )

    model = joblib.load(
        model_path
    )

    print(
        f"\nBest model: {best_model_name}"
    )

    print(
        f"Model path: {model_path}"
    )

    return model, best_model_name



# LOAD ENCODED TEST DATA


def load_dataset():

    X_test_path = os.path.join(
        ENCODED_DIR,
        "X_test.csv"
    )

    y_test_path = os.path.join(
        ENCODED_DIR,
        "y_test.csv"
    )

    if not os.path.exists(X_test_path):

        raise FileNotFoundError(
            f"\nX_test.csv not found:\n"
            f"{X_test_path}\n\n"
            f"Run the ML preprocessing/training steps first."
        )

    if not os.path.exists(y_test_path):

        raise FileNotFoundError(
            f"\ny_test.csv not found:\n"
            f"{y_test_path}"
        )

    X_test = pd.read_csv(
        X_test_path
    )

    y_test = pd.read_csv(
        y_test_path
    )

    # Handle either a named target column
    # or a single unnamed column.

    if "is_cancelled" in y_test.columns:

        y_test = y_test[
            "is_cancelled"
        ]

    else:

        y_test = y_test.iloc[
            :,
            0
        ]

    return X_test, y_test



# CHECK MODEL FEATURES


def validate_features(
    model,
    X_test
):

    print(
        "\nChecking model features..."
    )

    # --------------------------------------------------------
    # Models trained directly on DataFrames usually expose
    # feature_names_in_.
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        expected_features = list(
            model.feature_names_in_
        )

        actual_features = list(
            X_test.columns
        )

        print(
            f"Model expects : "
            f"{len(expected_features)} features"
        )

        print(
            f"Test data has : "
            f"{len(actual_features)} features"
        )

        missing = [
            feature
            for feature in expected_features
            if feature not in actual_features
        ]

        extra = [
            feature
            for feature in actual_features
            if feature not in expected_features
        ]

        if missing:

            print(
                "\nMissing features:"
            )

            for feature in missing:
                print(
                    f"  - {feature}"
                )

            raise ValueError(
                "\nThe test dataset is missing "
                "features required by the model."
            )

        if extra:

            print(
                "\nExtra features detected:"
            )

            for feature in extra:
                print(
                    f"  - {feature}"
                )

            # Keep only model features
            X_test = X_test[
                expected_features
            ]

        else:

            # Ensure exact training order
            X_test = X_test[
                expected_features
            ]

    print(
        "Feature validation: PASSED"
    )

    return X_test



# GENERATE RISK LEVEL


def generate_risk_level(
    probability
):

    if probability < 0.30:

        return "LOW"

    elif probability < 0.60:

        return "MEDIUM"

    else:

        return "HIGH"



# GENERATE CANCELLATION RISK


def generate_risk_scores(
    model,
    X_test,
    y_test
):

    print(
        "\nGenerating cancellation probabilities..."
    )

    # --------------------------------------------------------
    # Probability of class 1 = cancellation
    # --------------------------------------------------------

    probabilities = (
        model
        .predict_proba(X_test)
        [:, 1]
    )

    # --------------------------------------------------------
    # Standard prediction threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # --------------------------------------------------------
    # Build result dataset
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "actual_is_cancelled":
                y_test.values,

            "cancellation_probability":
                probabilities,

            "predicted_cancellation":
                predictions
        }
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    result[
        "risk_level"
    ] = result[
        "cancellation_probability"
    ].apply(
        generate_risk_level
    )

    return result



# DISPLAY SUMMARY


def display_summary(
    result
):

    print("\n")
    
    print("CANCELLATION RISK SUMMARY")
    

    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    risk_counts = (
        result[
            "risk_level"
        ]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH"
            ],
            fill_value=0
        )
    )

    print(
        f"\nLOW RISK    : "
        f"{risk_counts['LOW']:,}"
    )

    print(
        f"MEDIUM RISK : "
        f"{risk_counts['MEDIUM']:,}"
    )

    print(
        f"HIGH RISK   : "
        f"{risk_counts['HIGH']:,}"
    )

    # --------------------------------------------------------
    # Average probability
    # --------------------------------------------------------

    average_probability = (
        result[
            "cancellation_probability"
        ].mean()
    )

    print(
        "\nAverage cancellation probability:"
    )

    print(
        f"{average_probability:.2%}"
    )

    # --------------------------------------------------------
    # Actual cancellation rate
    # --------------------------------------------------------

    actual_rate = (
        result[
            "actual_is_cancelled"
        ].mean()
    )

    print(
        "\nActual cancellation rate:"
    )

    print(
        f"{actual_rate:.2%}"
    )

    # --------------------------------------------------------
    # Predicted cancellation rate
    # --------------------------------------------------------

    predicted_rate = (
        result[
            "predicted_cancellation"
        ].mean()
    )

    print(
        "\nPredicted cancellation rate:"
    )

    print(
        f"{predicted_rate:.2%}"
    )

    # --------------------------------------------------------
    # Highest risk bookings
    # --------------------------------------------------------

    print(
        "\nHighest-risk bookings:"
    )

    top_risk = (
        result
        .sort_values(
            "cancellation_probability",
            ascending=False
        )
        .head(10)
    )

    print(
        top_risk.to_string(
            index=False
        )
    )



# SAVE RESULTS


def save_results(
    result
):

    output_path = os.path.join(
        REPORT_DIR,
        "cancellation_risk_scores.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        "\nRisk scores saved:"
    )

    print(
        output_path
    )



# MAIN


def main():

    
    print("STEP 10.7 — CANCELLATION RISK SCORING")
    

    # --------------------------------------------------------
    # 1. Load best model
    # --------------------------------------------------------

    model, model_name = (
        load_best_model()
    )

    # --------------------------------------------------------
    # 2. Load encoded test data
    # --------------------------------------------------------

    X_test, y_test = (
        load_dataset()
    )

    print(
        f"\nTest bookings analyzed: "
        f"{len(X_test):,}"
    )

    print(
        f"Feature count: "
        f"{X_test.shape[1]}"
    )

    # --------------------------------------------------------
    # 3. Validate feature compatibility
    # --------------------------------------------------------

    X_test = validate_features(
        model,
        X_test
    )

    # --------------------------------------------------------
    # 4. Generate risk scores
    # --------------------------------------------------------

    result = generate_risk_scores(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # 5. Display summary
    # --------------------------------------------------------

    display_summary(
        result
    )

    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------

    save_results(
        result
    )

    # --------------------------------------------------------
    # 7. Completion
    # --------------------------------------------------------

    print("\n")
    
    print("STEP 10.7 COMPLETED")
    



# ENTRY POINT


if __name__ == "__main__":

    main()