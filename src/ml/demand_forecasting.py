import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


warnings.filterwarnings("ignore")



# STEP 10.9
# DEMAND FORECASTING




# PROJECT PATHS


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bookings_clean.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "ml"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ml",
    "models"
)

OUTPUT_DATA = os.path.join(
    REPORT_DIR,
    "monthly_demand_forecast.csv"
)

OUTPUT_METRICS = os.path.join(
    REPORT_DIR,
    "demand_forecast_metrics.csv"
)

OUTPUT_PLOT = os.path.join(
    REPORT_DIR,
    "demand_forecast.png"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "demand_forecasting_random_forest.pkl"
)


os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# LOAD BOOKINGS


def load_bookings():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nBookings dataset not found:\n"
            f"{DATA_PATH}"
        )

    data = pd.read_csv(
        DATA_PATH
    )

    required_columns = [
        "booking_id",
        "booking_date"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["booking_date"] = pd.to_datetime(
        data["booking_date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["booking_date"]
    )

    print(
        f"\nBookings loaded: {len(data):,}"
    )

    print(
        f"Date range: "
        f"{data['booking_date'].min().date()} "
        f"to "
        f"{data['booking_date'].max().date()}"
    )

    return data



# CREATE MONTHLY DEMAND


def create_monthly_demand(
    data
):

    monthly = (
        data
        .set_index("booking_date")
        .resample("MS")
        .size()
        .reset_index(
            name="booking_count"
        )
    )

    # Make sure missing months are represented
    # as zero demand.

    full_range = pd.date_range(
        start=monthly["booking_date"].min(),
        end=monthly["booking_date"].max(),
        freq="MS"
    )

    monthly = (
        monthly
        .set_index("booking_date")
        .reindex(full_range)
        .fillna(0)
        .rename_axis("booking_date")
        .reset_index()
    )

    monthly["booking_count"] = (
        monthly["booking_count"]
        .astype(int)
    )

    print(
        f"\nMonthly observations: "
        f"{len(monthly)}"
    )

    print(
        "\nMonthly demand:"
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    return monthly



# CREATE FORECAST FEATURES


def create_features(
    monthly
):

    data = monthly.copy()

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["year"] = (
        data["booking_date"]
        .dt.year
    )

    data["month"] = (
        data["booking_date"]
        .dt.month
    )

    # Sequential time index
    data["time_index"] = np.arange(
        len(data)
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    data["lag_1"] = (
        data["booking_count"]
        .shift(1)
    )

    data["lag_2"] = (
        data["booking_count"]
        .shift(2)
    )

    data["lag_3"] = (
        data["booking_count"]
        .shift(3)
    )

    data["lag_6"] = (
        data["booking_count"]
        .shift(6)
    )

    data["lag_12"] = (
        data["booking_count"]
        .shift(12)
    )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    data["rolling_mean_3"] = (
        data["booking_count"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    data["rolling_mean_6"] = (
        data["booking_count"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    data["rolling_mean_12"] = (
        data["booking_count"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    data["rolling_std_6"] = (
        data["booking_count"]
        .shift(1)
        .rolling(6)
        .std()
    )

    # --------------------------------------------------------
    # Remove rows where lag information isn't available
    # --------------------------------------------------------

    data = data.dropna()

    return data



# TRAIN / TEST SPLIT


def split_data(
    data
):

    feature_columns = [
        "year",
        "month",
        "time_index",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_std_6"
    ]

    target_column = (
        "booking_count"
    )

    X = data[
        feature_columns
    ]

    y = data[
        target_column
    ]

    # --------------------------------------------------------
    # Time-series split
    #
    # NEVER randomly shuffle time-series data.
    # --------------------------------------------------------

    test_size = max(
        4,
        int(len(data) * 0.20)
    )

    train_size = (
        len(data) - test_size
    )

    X_train = X.iloc[
        :train_size
    ]

    X_test = X.iloc[
        train_size:
    ]

    y_train = y.iloc[
        :train_size
    ]

    y_test = y.iloc[
        train_size:
    ]

    test_dates = data[
        "booking_date"
    ].iloc[
        train_size:
    ]

    # print(
    #     "\n"
    #     + "=" * 80
    # )

    print(
        "TIME-SERIES TRAIN / TEST SPLIT"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTraining observations: "
        f"{len(X_train)}"
    )

    print(
        f"Testing observations : "
        f"{len(X_test)}"
    )

    print(
        f"\nTraining period:"
    )

    print(
        f"{data['booking_date'].iloc[0].date()} "
        f"to "
        f"{data['booking_date'].iloc[train_size - 1].date()}"
    )

    print(
        "\nTesting period:"
    )

    print(
        f"{test_dates.iloc[0].date()} "
        f"to "
        f"{test_dates.iloc[-1].date()}"
    )

    return (
        feature_columns,
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates
    )



# TRAIN MODEL


def train_model(
    X_train,
    y_train
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TRAINING DEMAND FORECASTING MODEL"
    )

    print(
        "=" * 80
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nRandom Forest model trained successfully."
    )

    return model



# EVALUATE MODEL


def evaluate_model(
    model,
    X_test,
    y_test,
    test_dates
):

    predictions = model.predict(
        X_test
    )

    # Demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # MAPE
    actual = np.array(
        y_test
    )

    predicted = np.array(
        predictions
    )

    non_zero = actual != 0

    if non_zero.any():

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero]
                    -
                    predicted[non_zero]
                )
                /
                actual[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DEMAND FORECAST MODEL EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nMAE  : {mae:.2f}"
    )

    print(
        f"RMSE : {rmse:.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    if not np.isnan(mape):

        print(
            f"MAPE : {mape:.2f}%"
        )

    # --------------------------------------------------------
    # Test predictions table
    # --------------------------------------------------------

    evaluation = pd.DataFrame(
        {
            "booking_date":
                test_dates.values,

            "actual_demand":
                y_test.values,

            "predicted_demand":
                np.round(
                    predictions,
                    2
                )
        }
    )

    evaluation["forecast_error"] = (
        evaluation["actual_demand"]
        -
        evaluation["predicted_demand"]
    )

    print(
        "\nTest-set predictions:"
    )

    print(
        evaluation.to_string(
            index=False
        )
    )

    metrics = pd.DataFrame(
        [
            {
                "model":
                    "Random Forest",

                "MAE":
                    mae,

                "RMSE":
                    rmse,

                "R2":
                    r2,

                "MAPE":
                    mape
            }
        ]
    )

    return (
        evaluation,
        metrics
    )



# FUTURE FORECAST


def forecast_future(
    model,
    monthly,
    feature_columns,
    periods=6
):

    history = monthly.copy()

    forecasts = []

    for _ in range(periods):

        next_date = (
            history["booking_date"].max()
            +
            pd.offsets.MonthBegin(1)
        )

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        year = next_date.year

        month = next_date.month

        time_index = len(history)

        # ----------------------------------------------------
        # Lag helper
        # ----------------------------------------------------

        values = (
            history["booking_count"]
            .tolist()
        )

        def lag(n):

            if len(values) >= n:

                return values[-n]

            return np.nan

        # ----------------------------------------------------
        # Rolling helpers
        # ----------------------------------------------------

        def rolling_mean(n):

            if len(values) >= n:

                return np.mean(
                    values[-n:]
                )

            return np.nan

        def rolling_std(n):

            if len(values) >= n:

                return np.std(
                    values[-n:],
                    ddof=1
                )

            return np.nan

        feature_row = {

            "year":
                year,

            "month":
                month,

            "time_index":
                time_index,

            "lag_1":
                lag(1),

            "lag_2":
                lag(2),

            "lag_3":
                lag(3),

            "lag_6":
                lag(6),

            "lag_12":
                lag(12),

            "rolling_mean_3":
                rolling_mean(3),

            "rolling_mean_6":
                rolling_mean(6),

            "rolling_mean_12":
                rolling_mean(12),

            "rolling_std_6":
                rolling_std(6)
        }

        X_future = pd.DataFrame(
            [feature_row]
        )

        X_future = X_future[
            feature_columns
        ]

        prediction = model.predict(
            X_future
        )[0]

        prediction = max(
            0,
            prediction
        )

        prediction = round(
            prediction,
            2
        )

        forecasts.append(
            {
                "booking_date":
                    next_date,

                "forecast_demand":
                    prediction
            }
        )

        # Add forecast to history so the next
        # forecast can use it as a lag.

        history = pd.concat(
            [
                history,

                pd.DataFrame(
                    [
                        {
                            "booking_date":
                                next_date,

                            "booking_count":
                                prediction
                        }
                    ]
                )
            ],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(
        forecasts
    )

    return forecast_df



# CREATE FINAL OUTPUT


def create_final_output(
    monthly,
    evaluation,
    future_forecast
):

    historical = monthly[
        [
            "booking_date",
            "booking_count"
        ]
    ].copy()

    historical = historical.rename(
        columns={
            "booking_count":
                "demand"
        }
    )

    historical["data_type"] = (
        "Historical"
    )

    test_output = evaluation[
        [
            "booking_date",
            "predicted_demand"
        ]
    ].copy()

    test_output = test_output.rename(
        columns={
            "predicted_demand":
                "demand"
        }
    )

    test_output["data_type"] = (
        "Test Forecast"
    )

    future_output = future_forecast.copy()

    future_output = future_output.rename(
        columns={
            "forecast_demand":
                "demand"
        }
    )

    future_output["data_type"] = (
        "Future Forecast"
    )

    result = pd.concat(
        [
            historical,
            test_output,
            future_output
        ],
        ignore_index=True
    )

    result = result.sort_values(
        "booking_date"
    )

    return result



# PLOT FORECAST


def plot_forecast(
    monthly,
    evaluation,
    future_forecast
):

    plt.figure(
        figsize=(14, 7)
    )

    # Historical demand

    plt.plot(
        monthly["booking_date"],
        monthly["booking_count"],
        marker="o",
        label="Historical Demand"
    )

    # Test predictions

    plt.plot(
        evaluation["booking_date"],
        evaluation["predicted_demand"],
        marker="o",
        linestyle="--",
        label="Test Forecast"
    )

    # Future forecast

    plt.plot(
        future_forecast["booking_date"],
        future_forecast["forecast_demand"],
        marker="o",
        linestyle="--",
        label="Future Forecast"
    )

    plt.title(
        "Monthly Ticket Booking Demand Forecast"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Number of Bookings"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150
    )

    plt.close()

    print(
        f"\nForecast chart saved:"
    )

    print(
        OUTPUT_PLOT
    )



# SAVE MODEL


def save_model(
    model
):

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nForecast model saved:"
    )

    print(
        MODEL_PATH
    )



# MAIN


def main():

    print("=" * 80)

    print(
        "STEP 10.9 — DEMAND FORECASTING"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    bookings = load_bookings()

    # --------------------------------------------------------
    # 2. Monthly aggregation
    # --------------------------------------------------------

    monthly = create_monthly_demand(
        bookings
    )

    # --------------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------------

    feature_data = create_features(
        monthly
    )

    print(
        f"\nForecast feature dataset:"
    )

    print(
        f"Rows    : {len(feature_data)}"
    )

    print(
        f"Features: {feature_data.shape[1] - 2}"
    )

    # --------------------------------------------------------
    # 4. Split data
    # --------------------------------------------------------

    (
        feature_columns,
        X_train,
        X_test,
        y_train,
        y_test,
        test_dates
    ) = split_data(
        feature_data
    )

    # --------------------------------------------------------
    # 5. Train model
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    (
        evaluation,
        metrics
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        test_dates
    )

    # --------------------------------------------------------
    # 7. Forecast next 6 months
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "NEXT 6 MONTH DEMAND FORECAST"
    )

    print(
        "=" * 80
    )

    future_forecast = forecast_future(
        model,
        monthly,
        feature_columns,
        periods=6
    )

    print(
        "\n"
        + future_forecast.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. Create final output
    # --------------------------------------------------------

    final_output = create_final_output(
        monthly,
        evaluation,
        future_forecast
    )

    # --------------------------------------------------------
    # 9. Save outputs
    # --------------------------------------------------------

    final_output.to_csv(
        OUTPUT_DATA,
        index=False
    )

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False
    )

    print(
        f"\nForecast dataset saved:"
    )

    print(
        OUTPUT_DATA
    )

    print(
        f"\nForecast metrics saved:"
    )

    print(
        OUTPUT_METRICS
    )

    # --------------------------------------------------------
    # 10. Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    # --------------------------------------------------------
    # 11. Create visualization
    # --------------------------------------------------------

    plot_forecast(
        monthly,
        evaluation,
        future_forecast
    )

    # --------------------------------------------------------
    # 12. Completion
    # --------------------------------------------------------

    print("\n")

    print("=" * 80)

    print(
        "STEP 10.9 COMPLETED"
    )

    print("=" * 80)



# ENTRY POINT


if __name__ == "__main__":

    main()