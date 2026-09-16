import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 11.1 — BUSINESS RECOMMENDATION ENGINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

CANCELLATION_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "cancellation_risk_scores.csv"
)

RFM_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "analytics",
    "rfm_customer_segments.csv"
)

CLV_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "analytics",
    "customer_lifetime_value.csv"
)

ROUTE_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "analytics",
    "route_demand_analysis.csv"
)

FORECAST_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "future_demand_analysis.csv"
)

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "recommendations"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "business_recommendations.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "recommendation_summary.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_existing_file(possible_paths):

    for path in possible_paths:

        if os.path.exists(path):
            return path

    return None


def load_optional_csv(path, name):

    if path is None:
        print(f"{name}: file not found - skipped")
        return None

    try:

        data = pd.read_csv(path)

        print(
            f"{name}: {len(data):,} rows loaded"
        )

        return data

    except Exception as exc:

        print(
            f"{name}: could not load file - {exc}"
        )

        return None


# ============================================================
# CANCELLATION RECOMMENDATIONS
# ============================================================

def cancellation_recommendations(data):

    recommendations = []

    if data is None or data.empty:
        return recommendations

    # Detect risk column
    risk_column = None

    for column in [
        "risk_level",
        "risk_category",
        "risk"
    ]:

        if column in data.columns:
            risk_column = column
            break

    probability_column = None

    for column in [
        "cancellation_probability",
        "risk_probability",
        "cancel_probability"
    ]:

        if column in data.columns:
            probability_column = column
            break

    if risk_column is None:
        return recommendations

    for _, row in data.iterrows():

        risk = str(
            row[risk_column]
        ).upper()

        probability = (
            row[probability_column]
            if probability_column
            else np.nan
        )

        booking_id = row.get(
            "booking_id",
            "UNKNOWN"
        )

        if risk == "HIGH":

            action = (
                "Send cancellation-prevention "
                "reminder and monitor booking."
            )

            priority = "HIGH"

        elif risk == "MEDIUM":

            action = (
                "Send travel reminder closer "
                "to departure."
            )

            priority = "MEDIUM"

        else:

            action = (
                "No immediate intervention required."
            )

            priority = "LOW"

        recommendations.append(
            {
                "recommendation_type":
                    "Cancellation Prevention",

                "entity_type":
                    "Booking",

                "entity_id":
                    booking_id,

                "priority":
                    priority,

                "risk_level":
                    risk,

                "probability":
                    probability,

                "recommendation":
                    action,

                "business_reason":
                    "Reduce preventable booking "
                    "cancellations."
            }
        )

    return recommendations


# ============================================================
# CUSTOMER RECOMMENDATIONS
# ============================================================

def customer_recommendations(data):

    recommendations = []

    if data is None or data.empty:
        return recommendations

    # Try to identify segment column
    segment_column = None

    for column in [
        "segment",
        "rfm_segment",
        "customer_segment"
    ]:

        if column in data.columns:
            segment_column = column
            break

    if segment_column is None:
        return recommendations

    customer_column = None

    for column in [
        "customer_id",
        "customer_unique_id"
    ]:

        if column in data.columns:
            customer_column = column
            break

    if customer_column is None:
        customer_column = data.columns[0]

    for _, row in data.iterrows():

        segment = str(
            row[segment_column]
        ).lower()

        customer_id = row[
            customer_column
        ]

        # ----------------------------------------------------
        # Champions
        # ----------------------------------------------------

        if "champion" in segment:

            priority = "HIGH"

            action = (
                "Offer loyalty rewards, "
                "premium benefits and referral incentives."
            )

            reason = (
                "High-value customers are likely "
                "to respond well to retention programs."
            )

        # ----------------------------------------------------
        # Loyal
        # ----------------------------------------------------

        elif "loyal" in segment:

            priority = "MEDIUM"

            action = (
                "Provide loyalty offers and "
                "personalized route promotions."
            )

            reason = (
                "Encourage continued repeat bookings."
            )

        # ----------------------------------------------------
        # At Risk
        # ----------------------------------------------------

        elif (
            "at risk" in segment
            or "risk" in segment
        ):

            priority = "HIGH"

            action = (
                "Launch win-back campaign "
                "with personalized discounts."
            )

            reason = (
                "Customer activity indicates "
                "potential churn risk."
            )

        # ----------------------------------------------------
        # New customers
        # ----------------------------------------------------

        elif (
            "new" in segment
            or "recent" in segment
        ):

            priority = "MEDIUM"

            action = (
                "Send onboarding offers and "
                "encourage a second booking."
            )

            reason = (
                "Converting first-time customers "
                "into repeat customers improves retention."
            )

        # ----------------------------------------------------
        # Others
        # ----------------------------------------------------

        else:

            priority = "LOW"

            action = (
                "Use targeted promotions based "
                "on customer booking behavior."
            )

            reason = (
                "Personalized engagement can "
                "increase booking frequency."
            )

        recommendations.append(
            {
                "recommendation_type":
                    "Customer Marketing",

                "entity_type":
                    "Customer",

                "entity_id":
                    customer_id,

                "priority":
                    priority,

                "risk_level":
                    segment,

                "probability":
                    np.nan,

                "recommendation":
                    action,

                "business_reason":
                    reason
            }
        )

    return recommendations


# ============================================================
# CLV RECOMMENDATIONS
# ============================================================

def clv_recommendations(data):

    recommendations = []

    if data is None or data.empty:
        return recommendations

    customer_column = None

    for column in [
        "customer_id",
        "customer_unique_id"
    ]:

        if column in data.columns:
            customer_column = column
            break

    if customer_column is None:
        return recommendations

    # Find CLV column
    clv_column = None

    for column in [
        "clv",
        "customer_lifetime_value",
        "estimated_clv",
        "predicted_clv"
    ]:

        if column in data.columns:
            clv_column = column
            break

    if clv_column is None:
        return recommendations

    numeric_clv = pd.to_numeric(
        data[clv_column],
        errors="coerce"
    )

    threshold_high = numeric_clv.quantile(
        0.75
    )

    threshold_low = numeric_clv.quantile(
        0.25
    )

    for index, row in data.iterrows():

        customer_id = row[
            customer_column
        ]

        clv = numeric_clv.loc[index]

        if pd.isna(clv):
            continue

        if clv >= threshold_high:

            priority = "HIGH"

            action = (
                "Prioritize customer for "
                "premium loyalty and retention programs."
            )

            reason = (
                "Customer belongs to the top CLV group."
            )

        elif clv <= threshold_low:

            priority = "LOW"

            action = (
                "Use cost-efficient targeted "
                "promotions to increase customer value."
            )

            reason = (
                "Customer currently has relatively "
                "low lifetime value."
            )

        else:

            priority = "MEDIUM"

            action = (
                "Encourage additional bookings "
                "through personalized offers."
            )

            reason = (
                "Customer has potential to increase "
                "lifetime value."
            )

        recommendations.append(
            {
                "recommendation_type":
                    "CLV Management",

                "entity_type":
                    "Customer",

                "entity_id":
                    customer_id,

                "priority":
                    priority,

                "risk_level":
                    "N/A",

                "probability":
                    np.nan,

                "recommendation":
                    action,

                "business_reason":
                    reason
            }
        )

    return recommendations


# ============================================================
# ROUTE RECOMMENDATIONS
# ============================================================

def route_recommendations(data):

    recommendations = []

    if data is None or data.empty:
        return recommendations

    route_column = None

    for column in [
        "route_id",
        "route_name",
        "route"
    ]:

        if column in data.columns:
            route_column = column
            break

    if route_column is None:
        return recommendations

    # Find demand column
    demand_column = None

    for column in [
        "booking_count",
        "bookings",
        "total_bookings",
        "demand"
    ]:

        if column in data.columns:
            demand_column = column
            break

    if demand_column is None:
        return recommendations

    numeric_demand = pd.to_numeric(
        data[demand_column],
        errors="coerce"
    )

    high_threshold = numeric_demand.quantile(
        0.75
    )

    low_threshold = numeric_demand.quantile(
        0.25
    )

    for index, row in data.iterrows():

        route = row[
            route_column
        ]

        demand = numeric_demand.loc[index]

        if pd.isna(demand):
            continue

        if demand >= high_threshold:

            priority = "HIGH"

            action = (
                "Increase capacity and monitor "
                "seat availability for this route."
            )

            reason = (
                "Route demand is in the highest "
                "demand quartile."
            )

        elif demand <= low_threshold:

            priority = "LOW"

            action = (
                "Consider promotions and "
                "capacity optimization."
            )

            reason = (
                "Route demand is relatively low."
            )

        else:

            priority = "MEDIUM"

            action = (
                "Maintain current capacity and "
                "monitor booking trends."
            )

            reason = (
                "Route demand is within the "
                "normal range."
            )

        recommendations.append(
            {
                "recommendation_type":
                    "Route Capacity",

                "entity_type":
                    "Route",

                "entity_id":
                    route,

                "priority":
                    priority,

                "risk_level":
                    "N/A",

                "probability":
                    np.nan,

                "recommendation":
                    action,

                "business_reason":
                    reason
            }
        )

    return recommendations


# ============================================================
# FORECAST RECOMMENDATIONS
# ============================================================

def forecast_recommendations(data):

    recommendations = []

    if data is None or data.empty:
        return recommendations

    demand_column = None

    for column in [
        "forecast_demand",
        "demand"
    ]:

        if column in data.columns:
            demand_column = column
            break

    if demand_column is None:
        return recommendations

    date_column = None

    for column in [
        "month",
        "booking_date"
    ]:

        if column in data.columns:
            date_column = column
            break

    if date_column is None:
        return recommendations

    numeric_demand = pd.to_numeric(
        data[demand_column],
        errors="coerce"
    )

    average = numeric_demand.mean()

    high_threshold = numeric_demand.quantile(
        0.75
    )

    low_threshold = numeric_demand.quantile(
        0.25
    )

    for index, row in data.iterrows():

        demand = numeric_demand.loc[index]

        if pd.isna(demand):
            continue

        period = row[
            date_column
        ]

        if demand >= high_threshold:

            priority = "HIGH"

            action = (
                "Prepare additional vehicles, "
                "seats and operational capacity."
            )

            reason = (
                "Forecast demand is above the "
                "75th percentile."
            )

        elif demand <= low_threshold:

            priority = "LOW"

            action = (
                "Consider targeted promotions "
                "and flexible capacity allocation."
            )

            reason = (
                "Forecast demand is relatively low."
            )

        else:

            priority = "MEDIUM"

            action = (
                "Maintain normal capacity and "
                "monitor bookings."
            )

            reason = (
                "Forecast demand is close to "
                "the expected average."
            )

        recommendations.append(
            {
                "recommendation_type":
                    "Demand Forecast",

                "entity_type":
                    "Month",

                "entity_id":
                    period,

                "priority":
                    priority,

                "risk_level":
                    "N/A",

                "probability":
                    np.nan,

                "recommendation":
                    action,

                "business_reason":
                    reason
            }
        )

    return recommendations


# ============================================================
# SUMMARY
# ============================================================

def create_summary(recommendations):

    # Accept either a list or a pandas DataFrame
    if recommendations is None:
        return pd.DataFrame()

    if isinstance(recommendations, pd.DataFrame):

        if recommendations.empty:
            return pd.DataFrame()

        df = recommendations.copy()

    else:

        if len(recommendations) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(recommendations)

    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(
            [
                "recommendation_type",
                "priority"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="recommendation_count"
        )
        .sort_values(
            [
                "recommendation_type",
                "priority"
            ]
        )
    )

    return summary

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 11.1 — BUSINESS RECOMMENDATION ENGINE"
    )
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Locate files
    # --------------------------------------------------------

    print("\nLoading analytics outputs...\n")

    cancellation = load_optional_csv(
        find_existing_file(
            [
                CANCELLATION_PATH
            ]
        ),
        "Cancellation Risk"
    )

    rfm = load_optional_csv(
        find_existing_file(
            [
                RFM_PATH
            ]
        ),
        "RFM Segmentation"
    )

    clv = load_optional_csv(
        find_existing_file(
            [
                CLV_PATH
            ]
        ),
        "CLV Analytics"
    )

    route = load_optional_csv(
        find_existing_file(
            [
                ROUTE_PATH
            ]
        ),
        "Route Analytics"
    )

    forecast = load_optional_csv(
        find_existing_file(
            [
                FORECAST_PATH
            ]
        ),
        "Demand Forecast"
    )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = []

    recommendations.extend(
        cancellation_recommendations(
            cancellation
        )
    )

    recommendations.extend(
        customer_recommendations(
            rfm
        )
    )

    recommendations.extend(
        clv_recommendations(
            clv
        )
    )

    recommendations.extend(
        route_recommendations(
            route
        )
    )

    recommendations.extend(
        forecast_recommendations(
            forecast
        )
    )

    # --------------------------------------------------------
    # Save recommendations
    # --------------------------------------------------------

    recommendations_df = pd.DataFrame(
        recommendations
    )

    if recommendations_df.empty:

        print(
            "\nNo recommendations generated."
        )

        return

    # Priority ordering

    priority_order = {
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3
    }

    recommendations_df[
        "priority_order"
    ] = recommendations_df[
        "priority"
    ].map(
        priority_order
    ).fillna(99)

    recommendations_df = (
        recommendations_df
        .sort_values(
            "priority_order"
        )
        .drop(
            columns=[
                "priority_order"
            ]
        )
    )

    recommendations_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_summary(
        recommendations_df
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RECOMMENDATION SUMMARY")
    print("=" * 80)

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Total recommendations: "
        f"{len(recommendations_df):,}"
    )

    print(
        f"High priority: "
        f"{(
            recommendations_df['priority']
            == 'HIGH'
        ).sum():,}"
    )

    print(
        f"Medium priority: "
        f"{(
            recommendations_df['priority']
            == 'MEDIUM'
        ).sum():,}"
    )

    print(
        f"Low priority: "
        f"{(
            recommendations_df['priority']
            == 'LOW'
        ).sum():,}"
    )

    print("\n")
    print(
        "Recommendations saved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSummary saved:"
    )

    print(
        SUMMARY_PATH
    )

    print("\n")
    print("=" * 80)
    print("STEP 11.1 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()