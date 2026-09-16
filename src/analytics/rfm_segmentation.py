import os
import pandas as pd



# CONFIGURATION


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "analytics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# LOAD DATA


def load_data():

    customers = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "customers_clean.csv"
        )
    )

    bookings = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "bookings_clean.csv"
        )
    )

    bookings["booking_date"] = pd.to_datetime(
        bookings["booking_date"]
    )

    bookings["travel_date"] = pd.to_datetime(
        bookings["travel_date"]
    )

    return customers, bookings



# PREPARE BOOKING DATA


def prepare_bookings(bookings):

    # --------------------------------------------------------
    # For RFM, use completed/confirmed bookings.
    # Cancelled bookings should not contribute to customer
    # monetary value.
    # --------------------------------------------------------

    if "booking_status" in bookings.columns:

        confirmed = bookings[
            bookings["booking_status"]
            .astype(str)
            .str.upper()
            .eq("CONFIRMED")
        ].copy()

    else:

        confirmed = bookings.copy()

    return confirmed



# CALCULATE RFM


def calculate_rfm(
    customers,
    bookings
):

    data = prepare_bookings(
        bookings
    )

    # --------------------------------------------------------
    # Reference date
    # --------------------------------------------------------
    #
    # We use the latest booking date in the dataset + 1 day.
    #
    # This makes the calculation independent of today's
    # real-world date.
    #
    # Example:
    #
    # Latest booking = 2026-08-31
    # Reference date  = 2026-09-01
    #
    # --------------------------------------------------------

    reference_date = (
        data["booking_date"].max()
        +
        pd.Timedelta(days=1)
    )

    # --------------------------------------------------------
    # RFM calculation
    # --------------------------------------------------------

    rfm = (
        data
        .groupby("customer_id")
        .agg(
            last_booking_date=(
                "booking_date",
                "max"
            ),

            frequency=(
                "booking_id",
                "count"
            ),

            monetary=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Recency
    # --------------------------------------------------------

    rfm["recency"] = (
        reference_date
        -
        rfm["last_booking_date"]
    ).dt.days

    # --------------------------------------------------------
    # Join customer information
    # --------------------------------------------------------

    customer_columns = [
        "customer_id",
        "customer_name",
        "gender",
        "age",
        "city"
    ]

    available_columns = [
        column
        for column in customer_columns
        if column in customers.columns
    ]

    rfm = rfm.merge(
        customers[available_columns],
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    first_columns = [
        "customer_id",
        "customer_name",
        "city",
        "last_booking_date",
        "recency",
        "frequency",
        "monetary"
    ]

    remaining_columns = [
        column
        for column in rfm.columns
        if column not in first_columns
    ]

    rfm = rfm[
        first_columns +
        remaining_columns
    ]

    return rfm, reference_date



# CREATE RFM SCORES


def create_rfm_scores(rfm):

    # --------------------------------------------------------
    # RECENCY SCORE
    # --------------------------------------------------------
    #
    # Lower recency is better.
    #
    # Therefore qcut labels are reversed:
    #
    # lowest days → 5
    # highest days → 1
    #
    # --------------------------------------------------------

    rfm["R_score"] = pd.qcut(
        rfm["recency"]
        .rank(method="first"),
        q=5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # --------------------------------------------------------
    # FREQUENCY SCORE
    # --------------------------------------------------------

    rfm["F_score"] = pd.qcut(
        rfm["frequency"]
        .rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # --------------------------------------------------------
    # MONETARY SCORE
    # --------------------------------------------------------

    rfm["M_score"] = pd.qcut(
        rfm["monetary"]
        .rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # --------------------------------------------------------
    # TOTAL RFM SCORE
    # --------------------------------------------------------

    rfm["RFM_score"] = (
        rfm["R_score"]
        +
        rfm["F_score"]
        +
        rfm["M_score"]
    )

    # --------------------------------------------------------
    # RFM CODE
    # --------------------------------------------------------

    rfm["RFM_code"] = (
        rfm["R_score"].astype(str)
        +
        rfm["F_score"].astype(str)
        +
        rfm["M_score"].astype(str)
    )

    return rfm



# CUSTOMER SEGMENTS


def assign_segments(rfm):

    def segment(row):

        r = row["R_score"]
        f = row["F_score"]
        m = row["M_score"]
        total = row["RFM_score"]

        # ----------------------------------------------------
        # CHAMPIONS
        # ----------------------------------------------------

        if (
            r >= 4
            and f >= 4
            and m >= 4
        ):
            return "Champions"

        # ----------------------------------------------------
        # LOYAL CUSTOMERS
        # ----------------------------------------------------

        if (
            f >= 4
            and r >= 3
        ):
            return "Loyal Customers"

        # ----------------------------------------------------
        # BIG SPENDERS
        # ----------------------------------------------------

        if (
            m >= 4
            and r >= 3
        ):
            return "Big Spenders"

        # ----------------------------------------------------
        # POTENTIAL LOYALISTS
        # ----------------------------------------------------

        if (
            r >= 4
            and f >= 2
            and m >= 2
        ):
            return "Potential Loyalists"

        # ----------------------------------------------------
        # NEW CUSTOMERS
        # ----------------------------------------------------

        if (
            r >= 4
            and f <= 2
        ):
            return "New Customers"

        # ----------------------------------------------------
        # AT RISK
        # ----------------------------------------------------

        if (
            r <= 2
            and f >= 3
        ):
            return "At Risk"

        # ----------------------------------------------------
        # CANNOT LOSE
        # ----------------------------------------------------

        if (
            r <= 2
            and f >= 4
            and m >= 4
        ):
            return "Cannot Lose Them"

        # ----------------------------------------------------
        # LOST CUSTOMERS
        # ----------------------------------------------------

        if (
            r == 1
            and f <= 2
        ):
            return "Lost Customers"

        # ----------------------------------------------------
        # OTHERS
        # ----------------------------------------------------

        if total >= 10:
            return "Promising Customers"

        return "Need Attention"

    rfm["segment"] = rfm.apply(
        segment,
        axis=1
    )

    return rfm



# SEGMENT SUMMARY


def segment_summary(rfm):

    summary = (
        rfm
        .groupby("segment")
        .agg(
            customers=(
                "customer_id",
                "count"
            ),

            total_bookings=(
                "frequency",
                "sum"
            ),

            total_revenue=(
                "monetary",
                "sum"
            ),

            average_recency=(
                "recency",
                "mean"
            ),

            average_frequency=(
                "frequency",
                "mean"
            ),

            average_monetary=(
                "monetary",
                "mean"
            ),

            average_rfm_score=(
                "RFM_score",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "total_revenue",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Revenue percentage
    # --------------------------------------------------------

    total_revenue = (
        summary["total_revenue"]
        .sum()
    )

    summary["revenue_percentage"] = (
        summary["total_revenue"]
        /
        total_revenue
        *
        100
    )

    return summary



# BUSINESS RECOMMENDATIONS


def generate_recommendations(
    summary
):

    print("\n")
    
    print("RFM BUSINESS RECOMMENDATIONS")
    

    for _, row in summary.iterrows():

        segment = row["segment"]

        customers = int(
            row["customers"]
        )

        revenue_percentage = (
            row["revenue_percentage"]
        )

        print("\n")
        print(f"SEGMENT: {segment}")
        print("-" * 60)

        print(
            f"Customers       : {customers:,}"
        )

        print(
            f"Revenue Share   : "
            f"{revenue_percentage:.2f}%"
        )

        if segment == "Champions":

            print(
                "Action: Reward these high-value "
                "customers with loyalty benefits, "
                "priority offers and exclusive deals."
            )

        elif segment == "Loyal Customers":

            print(
                "Action: Encourage continued bookings "
                "through loyalty programs and "
                "personalized offers."
            )

        elif segment == "Big Spenders":

            print(
                "Action: Promote premium routes, "
                "premium vehicles and higher-value "
                "travel packages."
            )

        elif segment == "Potential Loyalists":

            print(
                "Action: Encourage the next booking "
                "with targeted discounts and "
                "membership benefits."
            )

        elif segment == "New Customers":

            print(
                "Action: Provide onboarding offers "
                "and incentives for a second booking."
            )

        elif segment == "At Risk":

            print(
                "Action: Launch re-engagement campaigns "
                "and identify reasons for reduced activity."
            )

        elif segment == "Cannot Lose Them":

            print(
                "Action: Prioritize retention because "
                "these customers have historically "
                "high value but are becoming inactive."
            )

        elif segment == "Lost Customers":

            print(
                "Action: Use win-back campaigns and "
                "limited-time offers to reactivate them."
            )

        elif segment == "Promising Customers":

            print(
                "Action: Increase engagement and "
                "encourage more frequent bookings."
            )

        else:

            print(
                "Action: Monitor behavior and develop "
                "targeted customer engagement campaigns."
            )



# SAVE RESULTS


def save_results(
    rfm,
    summary
):

    rfm_file = os.path.join(
        OUTPUT_DIR,
        "rfm_customer_segments.csv"
    )

    summary_file = os.path.join(
        OUTPUT_DIR,
        "rfm_segment_summary.csv"
    )

    rfm.to_csv(
        rfm_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n")
    
    print("FILES SAVED")
    

    print(
        f"RFM Customer Data:\n{rfm_file}"
    )

    print(
        f"\nSegment Summary:\n{summary_file}"
    )



# MAIN


def main():

    
    print("STEP 9.2 — RFM CUSTOMER SEGMENTATION")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    customers, bookings = load_data()

    print(
        f"\nCustomers loaded : {len(customers):,}"
    )

    print(
        f"Bookings loaded  : {len(bookings):,}"
    )

    # --------------------------------------------------------
    # Calculate RFM
    # --------------------------------------------------------

    print("\nCalculating RFM metrics...")

    rfm, reference_date = calculate_rfm(
        customers,
        bookings
    )

    print(
        f"Reference date   : "
        f"{reference_date.date()}"
    )

    print(
        f"Active customers : "
        f"{len(rfm):,}"
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    print(
        "\nCreating RFM scores..."
    )

    rfm = create_rfm_scores(
        rfm
    )

    # --------------------------------------------------------
    # Segments
    # --------------------------------------------------------

    print(
        "Assigning customer segments..."
    )

    rfm = assign_segments(
        rfm
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = segment_summary(
        rfm
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    
    print("RFM SEGMENT SUMMARY")
    

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top customers
    # --------------------------------------------------------

    print("\n")
    
    print("TOP 20 CUSTOMERS BY RFM SCORE")
    

    top_customers = (
        rfm
        .sort_values(
            [
                "RFM_score",
                "monetary"
            ],
            ascending=False
        )
        .head(20)
    )

    print(
        top_customers[
            [
                "customer_id",
                "customer_name",
                "city",
                "recency",
                "frequency",
                "monetary",
                "R_score",
                "F_score",
                "M_score",
                "RFM_score",
                "segment"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    generate_recommendations(
        summary
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        rfm,
        summary
    )

    print("\n")
    
    print("STEP 9.2 COMPLETED")
    


if __name__ == "__main__":
    main()