import os
import pandas as pd


# ============================================================
# STEP 11.6 — RECOMMENDATION CONSOLIDATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "recommendations"
)

OUTPUT_PATH = os.path.join(
    INPUT_DIR,
    "master_business_recommendations.csv"
)

SUMMARY_PATH = os.path.join(
    INPUT_DIR,
    "master_recommendation_summary.csv"
)


# ============================================================
# LOAD FILE
# ============================================================

def load_file(filename):

    path = os.path.join(
        INPUT_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            f"Skipped: {filename}"
        )

        return pd.DataFrame()

    data = pd.read_csv(
        path
    )

    print(
        f"{filename}: "
        f"{len(data):,} records"
    )

    return data


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

def normalize_columns(data):

    if data.empty:

        return data

    data = data.copy()

    required_columns = [
        "recommendation_type",
        "entity_type",
        "entity_id",
        "priority",
        "recommendation",
        "business_reason"
    ]

    for column in required_columns:

        if column not in data.columns:

            data[column] = ""

    return data


# ============================================================
# PRIORITY SCORE
# ============================================================

def add_priority_score(data):

    priority_map = {
        "URGENT": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }

    data["priority_score"] = (
        data["priority"]
        .astype(str)
        .str.upper()
        .map(priority_map)
        .fillna(0)
    )

    return data


# ============================================================
# SUMMARY
# ============================================================

def create_summary(data):

    if data.empty:

        return pd.DataFrame()

    summary = (
        data.groupby(
            [
                "recommendation_type",
                "entity_type",
                "priority"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="recommendation_count"
        )
    )

    return summary.sort_values(
        "recommendation_count",
        ascending=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 11.6 — MASTER RECOMMENDATION CONSOLIDATION"
    )
    print("=" * 80)

    os.makedirs(
        INPUT_DIR,
        exist_ok=True
    )

    datasets = []

    # --------------------------------------------------------
    # Existing engine
    # --------------------------------------------------------

    files = [
        "business_recommendations.csv",
        "cancellation_interventions.csv",
        "customer_marketing_recommendations.csv",
        "route_capacity_recommendations.csv",
        "demand_operations_recommendations.csv"
    ]

    print("\nLoading recommendation outputs...\n")

    for filename in files:

        data = load_file(
            filename
        )

        if not data.empty:

            data = normalize_columns(
                data
            )

            datasets.append(
                data
            )

    if not datasets:

        print(
            "\nNo recommendation datasets found."
        )

        return

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    master = pd.concat(
        datasets,
        ignore_index=True,
        sort=False
    )

    master = add_priority_score(
        master
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    master = master.drop_duplicates(
        subset=[
            "recommendation_type",
            "entity_type",
            "entity_id",
            "recommendation"
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    master = master.sort_values(
        [
            "priority_score",
            "entity_type"
        ],
        ascending=[
            False,
            True
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    master.to_csv(
        OUTPUT_PATH,
        index=False
    )

    summary = create_summary(
        master
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
    print(
        "MASTER RECOMMENDATION SUMMARY"
    )
    print("=" * 80)

    print(
        f"\nTotal recommendations: "
        f"{len(master):,}"
    )

    print("\nPriority distribution:")

    print(
        master[
            "priority"
        ]
        .value_counts()
        .to_string()
    )

    print("\nRecommendation categories:")

    print(
        master[
            "recommendation_type"
        ]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nEntity distribution:")

    print(
        master[
            "entity_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\nMaster file saved:")

    print(
        OUTPUT_PATH
    )

    print("\nSummary file saved:")

    print(
        SUMMARY_PATH
    )

    print("\n")
    print("=" * 80)
    print(
        "STEP 11.6 COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()