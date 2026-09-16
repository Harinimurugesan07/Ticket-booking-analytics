import os
import pandas as pd


# ============================================================
# STEP 11.5 — DEMAND & OPERATIONS RECOMMENDATIONS
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "recommendations"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "demand_operations_recommendations.csv"
)


# ============================================================
# LOAD FORECAST
# ============================================================

def load_forecast():

    data = pd.read_csv(
        FORECAST_PATH
    )

    data["booking_date"] = pd.to_datetime(
        data["booking_date"]
    )

    return data[
        data["data_type"]
        .eq("Future Forecast")
    ].copy()


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    forecast
):

    if forecast.empty:

        return pd.DataFrame()

    average_demand = (
        forecast["demand"]
        .mean()
    )

    recommendations = []

    for _, row in forecast.iterrows():

        demand = row[
            "demand"
        ]

        month = row[
            "booking_date"
        ].strftime(
            "%Y-%m"
        )

        ratio = (
            demand /
            average_demand
        )

        if ratio >= 1.10:

            priority = "HIGH"

            recommendation_type = (
                "Peak Capacity Planning"
            )

            recommendation = (
                "Increase vehicle availability, "
                "seat capacity and operator coverage."
            )

            reason = (
                "Forecast demand is significantly "
                "above the six-month average."
            )

        elif ratio <= 0.90:

            priority = "MEDIUM"

            recommendation_type = (
                "Low Demand Promotion"
            )

            recommendation = (
                "Consider targeted promotions and "
                "optimize vehicle allocation."
            )

            reason = (
                "Forecast demand is below the "
                "six-month average."
            )

        else:

            priority = "LOW"

            recommendation_type = (
                "Normal Capacity Planning"
            )

            recommendation = (
                "Maintain normal operational capacity "
                "and monitor demand."
            )

            reason = (
                "Forecast demand is close to the "
                "expected average."
            )

        recommendations.append(
            {
                "recommendation_type":
                    recommendation_type,

                "entity_type":
                    "Month",

                "entity_id":
                    month,

                "forecast_demand":
                    round(
                        demand,
                        2
                    ),

                "average_forecast_demand":
                    round(
                        average_demand,
                        2
                    ),

                "priority":
                    priority,

                "recommendation":
                    recommendation,

                "business_reason":
                    reason
            }
        )

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 11.5 — DEMAND & OPERATIONS RECOMMENDATIONS"
    )
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    forecast = load_forecast()

    print(
        f"\nFuture forecast months: "
        f"{len(forecast)}"
    )

    result = generate_recommendations(
        forecast
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print(
        result.to_string(
            index=False
        )
    )

    print("\nOutput saved:")

    print(
        OUTPUT_PATH
    )

    print("\n")
    print("=" * 80)
    print(
        "STEP 11.5 COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()