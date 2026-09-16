import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 10.10 — DEMAND FORECAST EVALUATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

FORECAST_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "monthly_demand_forecast.csv"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "demand_forecast_metrics.csv"
)

TEST_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "demand_test_evaluation.csv"
)

INSIGHTS_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "demand_business_insights.csv"
)

FUTURE_OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "future_demand_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(FORECAST_PATH):
        raise FileNotFoundError(
            f"Forecast file not found:\n{FORECAST_PATH}"
        )

    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics file not found:\n{METRICS_PATH}"
        )

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    metrics = pd.read_csv(
        METRICS_PATH
    )

    forecast["booking_date"] = pd.to_datetime(
        forecast["booking_date"]
    )

    return forecast, metrics


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def evaluate_model(metrics):

    print()
    print("=" * 80)
    print("FORECAST MODEL PERFORMANCE")
    print("=" * 80)

    row = metrics.iloc[0]

    model = row["model"]
    mae = float(row["MAE"])
    rmse = float(row["RMSE"])
    r2 = float(row["R2"])
    mape = float(row["MAPE"])

    print(f"\nModel : {model}")
    print(f"MAE   : {mae:.2f}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"R²    : {r2:.4f}")
    print(f"MAPE  : {mape:.2f}%")

    print("\nInterpretation:")

    if mape < 10:
        print("Excellent forecast accuracy.")

    elif mape < 20:
        print("Good forecast accuracy.")

    elif mape < 30:
        print("Acceptable forecast accuracy.")

    else:
        print(
            "High forecast error detected."
        )

    if r2 >= 0.70:
        print(
            "The model explains a strong proportion "
            "of demand variation."
        )

    elif r2 >= 0.40:
        print(
            "The model explains a moderate proportion "
            "of demand variation."
        )

    else:
        print(
            "The model has weak predictive performance "
            "on the current test period."
        )

    return {
        "model": model,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


# ============================================================
# TEST FORECAST EVALUATION
# ============================================================

def evaluate_test_forecast(forecast):

    print()
    print("=" * 80)
    print("TEST FORECAST ANALYSIS")
    print("=" * 80)

    historical = forecast[
        forecast["data_type"] == "Historical"
    ].copy()

    test_forecast = forecast[
        forecast["data_type"] == "Test Forecast"
    ].copy()

    if historical.empty or test_forecast.empty:

        print(
            "\nHistorical or Test Forecast data is missing."
        )

        return pd.DataFrame()

    historical = historical[
        [
            "booking_date",
            "demand"
        ]
    ].rename(
        columns={
            "demand": "actual_demand"
        }
    )

    test_forecast = test_forecast[
        [
            "booking_date",
            "demand"
        ]
    ].rename(
        columns={
            "demand": "predicted_demand"
        }
    )

    result = test_forecast.merge(
        historical,
        on="booking_date",
        how="inner"
    )

    result = result.sort_values(
        "booking_date"
    )

    result["forecast_error"] = (
        result["actual_demand"]
        -
        result["predicted_demand"]
    )

    result["absolute_error"] = (
        result["forecast_error"]
        .abs()
    )

    result["percentage_error"] = np.where(
        result["actual_demand"] != 0,
        (
            result["absolute_error"]
            /
            result["actual_demand"]
        ) * 100,
        np.nan
    )

    result["forecast_direction"] = np.where(
        result["forecast_error"] > 0,
        "Under Forecast",
        np.where(
            result["forecast_error"] < 0,
            "Over Forecast",
            "Accurate"
        )
    )

    print()

    print(
        result[
            [
                "booking_date",
                "actual_demand",
                "predicted_demand",
                "forecast_error",
                "percentage_error",
                "forecast_direction"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_actual = result[
        "actual_demand"
    ].sum()

    total_predicted = result[
        "predicted_demand"
    ].sum()

    total_difference = (
        total_actual
        -
        total_predicted
    )

    print()
    print("-" * 80)
    print("TEST PERIOD SUMMARY")
    print("-" * 80)

    print(
        f"Actual demand     : {total_actual:.0f}"
    )

    print(
        f"Predicted demand  : {total_predicted:.0f}"
    )

    print(
        f"Difference        : {total_difference:.0f}"
    )

    under_forecast = (
        result["forecast_direction"]
        == "Under Forecast"
    ).sum()

    over_forecast = (
        result["forecast_direction"]
        == "Over Forecast"
    ).sum()

    print(
        f"Under-forecast months : {under_forecast}"
    )

    print(
        f"Over-forecast months  : {over_forecast}"
    )

    result.to_csv(
        TEST_OUTPUT_PATH,
        index=False
    )

    print(
        f"\nTest evaluation saved:"
    )

    print(
        TEST_OUTPUT_PATH
    )

    return result


# ============================================================
# FUTURE DEMAND ANALYSIS
# ============================================================

def analyze_future_demand(forecast):

    print()
    print("=" * 80)
    print("FUTURE DEMAND ANALYSIS")
    print("=" * 80)

    future = forecast[
        forecast["data_type"] == "Future Forecast"
    ].copy()

    if future.empty:

        print(
            "\nNo Future Forecast records found."
        )

        return pd.DataFrame()

    future = future.sort_values(
        "booking_date"
    )

    future["month"] = future[
        "booking_date"
    ].dt.strftime("%Y-%m")

    future["forecast_demand"] = future[
        "demand"
    ]

    print()

    print(
        future[
            [
                "month",
                "forecast_demand"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_demand = future[
        "forecast_demand"
    ].sum()

    average_demand = future[
        "forecast_demand"
    ].mean()

    maximum_demand = future[
        "forecast_demand"
    ].max()

    minimum_demand = future[
        "forecast_demand"
    ].min()

    max_index = future[
        "forecast_demand"
    ].idxmax()

    min_index = future[
        "forecast_demand"
    ].idxmin()

    peak_month = future.loc[
        max_index,
        "month"
    ]

    low_month = future.loc[
        min_index,
        "month"
    ]

    print()
    print("-" * 80)
    print("FORECAST SUMMARY")
    print("-" * 80)

    print(
        f"Forecast months       : {len(future)}"
    )

    print(
        f"Expected total demand : {total_demand:.0f}"
    )

    print(
        f"Average monthly demand: {average_demand:.0f}"
    )

    print(
        f"Peak demand           : {maximum_demand:.0f}"
    )

    print(
        f"Peak month            : {peak_month}"
    )

    print(
        f"Lowest demand         : {minimum_demand:.0f}"
    )

    print(
        f"Lowest month          : {low_month}"
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    first_value = future.iloc[0][
        "forecast_demand"
    ]

    last_value = future.iloc[-1][
        "forecast_demand"
    ]

    if first_value != 0:

        trend_change = (
            (
                last_value
                -
                first_value
            )
            /
            first_value
        ) * 100

    else:

        trend_change = 0

    if trend_change > 10:

        trend = "Increasing"

    elif trend_change < -10:

        trend = "Decreasing"

    else:

        trend = "Stable"

    print(
        f"\nDemand trend: {trend}"
    )

    print(
        f"Change from first to last "
        f"forecast month: {trend_change:.2f}%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    future_output = future[
        [
            "booking_date",
            "month",
            "forecast_demand"
        ]
    ].copy()

    future_output["trend"] = trend

    future_output.to_csv(
        FUTURE_OUTPUT_PATH,
        index=False
    )

    print(
        f"\nFuture demand analysis saved:"
    )

    print(
        FUTURE_OUTPUT_PATH
    )

    return future_output


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

def generate_business_insights(
    model_results,
    test_results,
    future_results
):

    print()
    print("=" * 80)
    print("BUSINESS INSIGHTS")
    print("=" * 80)

    insights = []

    # --------------------------------------------------------
    # Model quality
    # --------------------------------------------------------

    mape = model_results["MAPE"]
    r2 = model_results["R2"]

    if mape < 20:

        model_message = (
            "Forecast accuracy is suitable "
            "for operational planning."
        )

    elif mape < 30:

        model_message = (
            "Forecast accuracy is acceptable, "
            "but forecasts should be monitored."
        )

    else:

        model_message = (
            "Current forecast accuracy is weak. "
            "Use forecasts as directional guidance "
            "rather than exact booking estimates."
        )

    insights.append(
        {
            "category": "Model Performance",
            "metric": "MAPE",
            "value": round(mape, 2),
            "insight": model_message
        }
    )

    insights.append(
        {
            "category": "Model Performance",
            "metric": "R2",
            "value": round(r2, 4),
            "insight":
                "Low or negative R² indicates that "
                "the current model does not explain "
                "the test-period demand variation well."
        }
    )

    # --------------------------------------------------------
    # Test period
    # --------------------------------------------------------

    if not test_results.empty:

        under_count = (
            test_results[
                "forecast_direction"
            ]
            == "Under Forecast"
        ).sum()

        over_count = (
            test_results[
                "forecast_direction"
            ]
            == "Over Forecast"
        ).sum()

        if under_count > over_count:

            test_message = (
                "The model tends to underestimate "
                "demand during the test period. "
                "Additional capacity may be required."
            )

        elif over_count > under_count:

            test_message = (
                "The model tends to overestimate "
                "demand. Avoid allocating excessive "
                "capacity based only on this forecast."
            )

        else:

            test_message = (
                "The model has a balanced forecast "
                "direction across the test months."
            )

        insights.append(
            {
                "category": "Forecast Bias",
                "metric": "Test Forecast Direction",
                "value":
                    f"Under={under_count}, "
                    f"Over={over_count}",
                "insight": test_message
            }
        )

    # --------------------------------------------------------
    # Future demand
    # --------------------------------------------------------

    if not future_results.empty:

        peak_row = future_results.loc[
            future_results[
                "forecast_demand"
            ].idxmax()
        ]

        low_row = future_results.loc[
            future_results[
                "forecast_demand"
            ].idxmin()
        ]

        average_demand = future_results[
            "forecast_demand"
        ].mean()

        peak_demand = peak_row[
            "forecast_demand"
        ]

        low_demand = low_row[
            "forecast_demand"
        ]

        peak_month = peak_row[
            "month"
        ]

        low_month = low_row[
            "month"
        ]

        insights.append(
            {
                "category": "Capacity Planning",
                "metric": "Peak Demand",
                "value": round(peak_demand, 2),
                "insight":
                    f"Highest forecast demand is "
                    f"{peak_demand:.0f} bookings in "
                    f"{peak_month}. Prioritize vehicle, "
                    f"seat and operator capacity."
            }
        )

        insights.append(
            {
                "category": "Capacity Planning",
                "metric": "Lowest Demand",
                "value": round(low_demand, 2),
                "insight":
                    f"Lowest forecast demand is "
                    f"{low_demand:.0f} bookings in "
                    f"{low_month}. Consider promotions "
                    f"or optimized capacity."
            }
        )

        insights.append(
            {
                "category": "Planning",
                "metric": "Average Monthly Demand",
                "value": round(average_demand, 2),
                "insight":
                    "Use average forecast demand as "
                    "a baseline for operational planning."
            }
        )

    # --------------------------------------------------------
    # Data quality warning
    # --------------------------------------------------------

    insights.append(
        {
            "category": "Data Quality",
            "metric": "Forecast Reliability",
            "value": "Review Required",
            "insight":
                "The source data contains a partially "
                "observed final month. Forecast evaluation "
                "can therefore be distorted. Complete "
                "months should be preferred for model "
                "training and evaluation."
        }
    )

    insights_df = pd.DataFrame(
        insights
    )

    print()

    print(
        insights_df.to_string(
            index=False
        )
    )

    insights_df.to_csv(
        INSIGHTS_OUTPUT_PATH,
        index=False
    )

    print(
        f"\nBusiness insights saved:"
    )

    print(
        INSIGHTS_OUTPUT_PATH
    )

    return insights_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 10.10 — FORECAST EVALUATION "
        "& BUSINESS INSIGHTS"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    forecast, metrics = load_data()

    print(
        f"\nForecast records loaded: "
        f"{len(forecast):,}"
    )

    print(
        f"Historical records: "
        f"{(forecast['data_type'] == 'Historical').sum()}"
    )

    print(
        f"Test forecast records: "
        f"{(forecast['data_type'] == 'Test Forecast').sum()}"
    )

    print(
        f"Future forecast records: "
        f"{(forecast['data_type'] == 'Future Forecast').sum()}"
    )

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    model_results = evaluate_model(
        metrics
    )

    # --------------------------------------------------------
    # Test forecast
    # --------------------------------------------------------

    test_results = evaluate_test_forecast(
        forecast
    )

    # --------------------------------------------------------
    # Future demand
    # --------------------------------------------------------

    future_results = analyze_future_demand(
        forecast
    )

    # --------------------------------------------------------
    # Business insights
    # --------------------------------------------------------

    generate_business_insights(
        model_results,
        test_results,
        future_results
    )

    print()
    print("=" * 80)
    print("STEP 10.10 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()