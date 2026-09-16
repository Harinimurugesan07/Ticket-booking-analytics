import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 11.2 — CANCELLATION INTERVENTION RECOMMENDATIONS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "ml",
    "cancellation_risk_scores.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "recommendations"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "cancellation_interventions.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "cancellation_intervention_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Cancellation risk file not found:\n"
            f"{INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    print(
        f"Cancellation risk records loaded: "
        f"{len(data):,}"
    )

    return data


# ============================================================
# FIND PROBABILITY COLUMN
# ============================================================

def find_probability_column(data):

    candidates = [
        "cancellation_probability",
        "cancel_probability",
        "risk_probability",
        "cancellation_risk",
        "probability"
    ]

    for column in candidates:

        if column in data.columns:
            return column

    raise ValueError(
        "Cancellation probability column not found.\n"
        f"Available columns: {data.columns.tolist()}"
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    # Probability may be stored as 0-1
    # or 0-100.

    if probability <= 1:

        probability = probability * 100

    if probability >= 80:

        return "CRITICAL"

    elif probability >= 60:

        return "HIGH"

    elif probability >= 30:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# INTERVENTION LOGIC
# ============================================================

def determine_intervention(risk):

    if risk == "CRITICAL":

        return {
            "priority": "URGENT",

            "intervention": (
                "Immediate retention intervention"
            ),

            "action": (
                "Contact customer immediately, "
                "send reminder and offer a targeted "
                "retention incentive."
            ),

            "objective": (
                "Prevent likely cancellation "
                "before travel."
            )
        }

    elif risk == "HIGH":

        return {
            "priority": "HIGH",

            "intervention": (
                "Cancellation prevention campaign"
            ),

            "action": (
                "Send personalized reminder and "
                "consider a targeted incentive."
            ),

            "objective": (
                "Reduce high-probability cancellations."
            )
        }

    elif risk == "MEDIUM":

        return {
            "priority": "MEDIUM",

            "intervention": (
                "Travel reminder"
            ),

            "action": (
                "Send automated reminder closer "
                "to travel date."
            ),

            "objective": (
                "Keep customer engaged and "
                "reduce avoidable cancellations."
            )
        }

    else:

        return {
            "priority": "LOW",

            "intervention": (
                "No immediate intervention"
            ),

            "action": (
                "Continue normal booking communication."
            ),

            "objective": (
                "Avoid unnecessary intervention cost."
            )
        }


# ============================================================
# GENERATE INTERVENTIONS
# ============================================================

def generate_interventions(data):

    probability_column = find_probability_column(
        data
    )

    data = data.copy()

    data["probability_numeric"] = pd.to_numeric(
        data[probability_column],
        errors="coerce"
    )

    data = data[
        data["probability_numeric"].notna()
    ].copy()

    # Normalize probability to percentage

    data["cancellation_probability_pct"] = np.where(
        data["probability_numeric"] <= 1,
        data["probability_numeric"] * 100,
        data["probability_numeric"]
    )

    data["risk_level"] = (
        data[
            "cancellation_probability_pct"
        ]
        .apply(classify_risk)
    )

    interventions = []

    for _, row in data.iterrows():

        risk = row["risk_level"]

        details = determine_intervention(
            risk
        )

        booking_id = row.get(
            "booking_id",
            "UNKNOWN"
        )

        customer_id = row.get(
            "customer_id",
            "UNKNOWN"
        )

        probability = row[
            "cancellation_probability_pct"
        ]

        intervention = {

            "booking_id":
                booking_id,

            "customer_id":
                customer_id,

            "cancellation_probability_pct":
                round(
                    probability,
                    2
                ),

            "risk_level":
                risk,

            "priority":
                details["priority"],

            "intervention_type":
                details["intervention"],

            "recommended_action":
                details["action"],

            "business_objective":
                details["objective"]
        }

        # Keep useful booking attributes
        for column in [
            "route_id",
            "vehicle_type",
            "seat_count",
            "payment_method",
            "total_amount",
            "lead_time_days"
        ]:

            if column in row.index:

                intervention[column] = row[
                    column
                ]

        interventions.append(
            intervention
        )

    return pd.DataFrame(
        interventions
    )


# ============================================================
# SUMMARY
# ============================================================

def create_summary(data):

    summary = (
        data.groupby(
            [
                "risk_level",
                "priority",
                "intervention_type"
            ]
        )
        .size()
        .reset_index(
            name="booking_count"
        )
    )

    risk_order = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4
    }

    summary["risk_order"] = (
        summary["risk_level"]
        .map(risk_order)
    )

    summary = (
        summary
        .sort_values(
            "risk_order"
        )
        .drop(
            columns=[
                "risk_order"
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
        "STEP 11.2 — CANCELLATION INTERVENTION RECOMMENDATIONS"
    )
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\nLoading cancellation risk data...\n")

    data = load_data()

    print("\nGenerating intervention recommendations...\n")

    result = generate_interventions(
        data
    )

    if result.empty:

        print(
            "No valid cancellation-risk records found."
        )

        return

    summary = create_summary(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("=" * 80)
    print(
        "CANCELLATION INTERVENTION SUMMARY"
    )
    print("=" * 80)

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Total bookings analyzed: "
        f"{len(result):,}"
    )

    print(
        f"Critical risk: "
        f"{(
            result['risk_level']
            == 'CRITICAL'
        ).sum():,}"
    )

    print(
        f"High risk: "
        f"{(
            result['risk_level']
            == 'HIGH'
        ).sum():,}"
    )

    print(
        f"Medium risk: "
        f"{(
            result['risk_level']
            == 'MEDIUM'
        ).sum():,}"
    )

    print(
        f"Low risk: "
        f"{(
            result['risk_level']
            == 'LOW'
        ).sum():,}"
    )

    print("\n")
    print(
        "Intervention report saved:"
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
    print(
        "STEP 11.2 COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()