import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



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



# LOAD DATA


def load_ml_data():

    filepath = os.path.join(
        ML_DIR,
        "cancellation_ml_dataset.csv"
    )

    data = pd.read_csv(filepath)

    return data



# PREPARE FEATURES


def prepare_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # Remove identifier columns
    # --------------------------------------------------------

    columns_to_remove = [
        "customer_id"
    ]

    data = data.drop(
        columns=[
            column
            for column in columns_to_remove
            if column in data.columns
        ]
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    X = data.drop(
        columns=["is_cancelled"]
    )

    y = data["is_cancelled"]

    return X, y



# IDENTIFY COLUMN TYPES


def identify_columns(X):

    categorical_features = [
        "payment_method",
        "route_id",
        "vehicle_type",
        "distance_category"
    ]

    categorical_features = [
        column
        for column in categorical_features
        if column in X.columns
    ]

    numerical_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    return numerical_features, categorical_features



# CREATE PREPROCESSOR


def create_preprocessor(
    numerical_features,
    categorical_features
):

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                "passthrough",
                numerical_features
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],

        remainder="drop"
    )

    return preprocessor



# TRAIN / TEST SPLIT


def split_data(
    X,
    y
):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



# MAIN


def main():

    
    print("STEP 10.2 — FEATURE ENGINEERING & ENCODING")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_ml_data()

    print(
        f"\nOriginal dataset shape: "
        f"{data.shape}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_features(
        data
    )

    print(
        f"\nFeatures shape: "
        f"{X.shape}"
    )

    print(
        f"Target shape: "
        f"{y.shape}"
    )

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    numerical_features, categorical_features = (
        identify_columns(X)
    )

    print("\nNumerical features:")
    print(
        numerical_features
    )

    print("\nCategorical features:")
    print(
        categorical_features
    )

    # --------------------------------------------------------
    # Create preprocessor
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    print("\nTrain/Test Split")
    print("-" * 80)

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    print(
        f"y_train : {y_train.shape}"
    )

    print(
        f"y_test  : {y_test.shape}"
    )

    # --------------------------------------------------------
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------------

    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_encoded = (
        preprocessor.transform(
            X_test
        )
    )

    print("\nEncoded data")
    print("-" * 80)

    print(
        f"X_train encoded : "
        f"{X_train_encoded.shape}"
    )

    print(
        f"X_test encoded  : "
        f"{X_test_encoded.shape}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nTraining target distribution")
    print("-" * 80)

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting target distribution")
    print("-" * 80)

    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print("\nEncoded feature count:")
    print(
        len(feature_names)
    )

    print("\nFirst 20 encoded features:")

    for feature in feature_names[:20]:

        print(
            f"  {feature}"
        )

    # --------------------------------------------------------
    # Save encoded arrays
    # --------------------------------------------------------

    encoded_dir = os.path.join(
        ML_DIR,
        "encoded"
    )

    os.makedirs(
        encoded_dir,
        exist_ok=True
    )

    pd.DataFrame(
        X_train_encoded,
        columns=feature_names
    ).to_csv(
        os.path.join(
            encoded_dir,
            "X_train.csv"
        ),
        index=False
    )

    pd.DataFrame(
        X_test_encoded,
        columns=feature_names
    ).to_csv(
        os.path.join(
            encoded_dir,
            "X_test.csv"
        ),
        index=False
    )

    pd.DataFrame(
        {
            "is_cancelled": y_train
        }
    ).to_csv(
        os.path.join(
            encoded_dir,
            "y_train.csv"
        ),
        index=False
    )

    pd.DataFrame(
        {
            "is_cancelled": y_test
        }
    ).to_csv(
        os.path.join(
            encoded_dir,
            "y_test.csv"
        ),
        index=False
    )

    print("\n")
    
    print("ENCODED DATA SAVED")
    

    print(
        os.path.join(
            encoded_dir,
            "X_train.csv"
        )
    )

    print(
        os.path.join(
            encoded_dir,
            "X_test.csv"
        )
    )

    print(
        os.path.join(
            encoded_dir,
            "y_train.csv"
        )
    )

    print(
        os.path.join(
            encoded_dir,
            "y_test.csv"
        )
    )

    print("\n")
    
    print("STEP 10.2 COMPLETED")
    


if __name__ == "__main__":
    main()