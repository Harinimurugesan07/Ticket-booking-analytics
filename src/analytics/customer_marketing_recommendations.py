import os
import glob
import pandas as pd
import numpy as np


# ============================================================
# STEP 11.3 — CUSTOMER MARKETING RECOMMENDATIONS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

OUTPUT_DIR = os.path.join(
    REPORTS_DIR,
    "recommendations"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "customer_marketing_recommendations.csv"
)


# ============================================================
# FIND RFM FILE
# ============================================================

def find_rfm_file():

    patterns = [
        "**/*rfm*.csv",
        "**/*customer*segment*.csv",
        "**/*segmentation*.csv"
    ]

    candidates = []

    for pattern in patterns:

        candidates.extend(
            glob.glob(
                os.path.join(
                    REPORTS_DIR,
                    pattern
                ),
                recursive=True
            )
        )

    # Remove recommendation files
    candidates = [
        p for p in candidates
        if "recommendation" not in p.lower()
    ]

    if not candidates:

        return None

    return candidates[0]


# ============================================================
# FIND CLV FILE
# ============================================================

def find_clv_file():

    patterns = [
        "**/*clv*.csv",
        "**/*lifetime*value*.csv",
        "**/*customer*value*.csv"
    ]

    candidates = []

    for pattern in patterns:

        candidates.extend(
            glob.glob(
                os.path.join(
                    REPORTS_DIR,
                    pattern
                ),
                recursive=True
            )
        )

    candidates = [
        p for p in candidates
        if "recommendation" not in p.lower()
    ]

    if not candidates:

        return None

    return candidates[0]


# ============================================================
# LOAD RFM
# ============================================================

def load_rfm():

    path = find_rfm_file()

    if path is None:

        print(
            "RFM file not found."
        )

        return pd.DataFrame()

    data = pd.read_csv(path)

    print(
        f"RFM file: {path}"
    )

    print(
        f"RFM records: {len(data):,}"
    )

    return data


# ============================================================
# LOAD CLV
# ============================================================

def load_clv():

    path = find_clv_file()

    if path is None:

        print(
            "CLV file not found - continuing "
            "without CLV."
        )

        return pd.DataFrame()

    data = pd.read_csv(path)

    print(
        f"CLV file: {path}"
    )

    print(
        f"CLV records: {len(data):,}"
    )

    return data


# ============================================================
# FIND SEGMENT COLUMN
# ============================================================

def find_segment_column(data):

    candidates = [
        "segment",
        "rfm_segment",
        "customer_segment",
        "segment_name"
    ]

    for column in candidates:

        if column in data.columns:
            return column

    return None


# ============================================================
# FIND CUSTOMER COLUMN
# ============================================================

def find_customer_column(data):

    candidates = [
        "customer_id",
        "customer_unique_id",
        "customer"
    ]

    for column in candidates:

        if column in data.columns:
            return column

    return None


# ============================================================
# GENERATE MARKETING RECOMMENDATION
# ============================================================

def recommendation_for_segment(segment):

    segment = str(
        segment
    ).strip().lower()

    if "champion" in segment:

        return (
            "Loyalty Rewards",
            "HIGH",
            "Offer loyalty rewards, premium benefits and referral incentives.",
            "High-value customers are likely to respond well to retention programs."
        )

    if "loyal" in segment:

        return (
            "Loyalty Campaign",
            "HIGH",
            "Provide loyalty points, member benefits and personalized offers.",
            "Repeated booking behavior indicates strong customer engagement."
        )

    if "at risk" in segment or "risk" in segment:

        return (
            "Win-back Campaign",
            "HIGH",
            "Launch a personalized win-back campaign with a targeted discount.",
            "Customer activity indicates potential churn risk."
        )

    if "lost" in segment or "hibernat" in segment:

        return (
            "Reactivation Campaign",
            "MEDIUM",
            "Send a reactivation offer and highlight relevant routes or services.",
            "Customer activity has declined significantly."
        )

    if "new" in segment:

        return (
            "Second Booking Campaign",
            "MEDIUM",
            "Offer a first-repeat booking incentive.",
            "The customer should be encouraged to make a second booking."
        )

    if "potential" in segment:

        return (
            "Engagement Campaign",
            "MEDIUM",
            "Provide personalized offers based on previous booking behavior.",
            "Customer behavior suggests potential for higher engagement."
        )

    if "promising" in segment:

        return (
            "Growth Campaign",
            "MEDIUM",
            "Promote relevant routes and loyalty benefits.",
            "Customer shows potential for increased booking frequency."
        )

    return (
        "General Engagement",
        "LOW",
        "Send personalized service updates and relevant offers.",
        "Maintain customer engagement without excessive marketing cost."
    )


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(rfm, clv):

    if rfm.empty:

        return pd.DataFrame()

    segment_column = find_segment_column(
        rfm
    )

    customer_column = find_customer_column(
        rfm
    )

    if segment_column is None:

        raise ValueError(
            "Could not find RFM segment column."
        )

    if customer_column is None:

        raise ValueError(
            "Could not find customer ID column."
        )

    data = rfm.copy()

    # --------------------------------------------------------
    # Merge CLV if possible
    # --------------------------------------------------------

    if not clv.empty:

        clv_customer_column = (
            find_customer_column(clv)
        )

        if clv_customer_column:

            data = data.merge(
                clv,
                left_on=customer_column,
                right_on=clv_customer_column,
                how="left",
                suffixes=(
                    "",
                    "_clv"
                )
            )

    recommendations = []

    for _, row in data.iterrows():

        segment = row[
            segment_column
        ]

        (
            recommendation_type,
            priority,
            recommendation,
            reason
        ) = recommendation_for_segment(
            segment
        )

        record = {

            "recommendation_type":
                recommendation_type,

            "entity_type":
                "Customer",

            "entity_id":
                row[customer_column],

            "customer_segment":
                segment,

            "priority":
                priority,

            "recommendation":
                recommendation,

            "business_reason":
                reason
        }

        # Include CLV when available
        for column in data.columns:

            if "clv" in column.lower():

                value = row[column]

                if pd.notna(value):

                    record[
                        "customer_lifetime_value"
                    ] = value

                    break

        recommendations.append(
            record
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
        "STEP 11.3 — CUSTOMER MARKETING RECOMMENDATIONS"
    )
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\nLoading customer analytics...\n")

    rfm = load_rfm()

    clv = load_clv()

    if rfm.empty:

        print(
            "\nNo RFM dataset available."
        )

        return

    print(
        "\nGenerating customer marketing recommendations...\n"
    )

    result = generate_recommendations(
        rfm,
        clv
    )

    if result.empty:

        print(
            "No recommendations generated."
        )

        return

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 80)
    print(
        "CUSTOMER MARKETING SUMMARY"
    )
    print("=" * 80)

    print()

    print(
        result[
            "recommendation_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\nPriority distribution:")

    print(
        result[
            "priority"
        ]
        .value_counts()
        .to_string()
    )

    print("\nOutput saved:")

    print(
        OUTPUT_PATH
    )

    print("\n")
    print("=" * 80)
    print(
        "STEP 11.3 COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()