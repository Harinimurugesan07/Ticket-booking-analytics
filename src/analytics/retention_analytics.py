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



# PREPARE CONFIRMED BOOKINGS


def prepare_bookings(bookings):

    data = bookings.copy()

    # Retention should be based on successful bookings.

    if "booking_status" in data.columns:

        data = data[
            data["booking_status"]
            .astype(str)
            .str.upper()
            .eq("CONFIRMED")
        ].copy()

    return data



# CUSTOMER BOOKING HISTORY


def customer_booking_history(bookings):

    history = (
        bookings
        .groupby("customer_id")
        .agg(
            first_booking_date=(
                "booking_date",
                "min"
            ),

            last_booking_date=(
                "booking_date",
                "max"
            ),

            total_bookings=(
                "booking_id",
                "count"
            ),

            total_tickets=(
                "seat_count",
                "sum"
            ),

            total_revenue=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Repeat customer
    # --------------------------------------------------------

    history["is_repeat_customer"] = (
        history["total_bookings"] > 1
    )

    # --------------------------------------------------------
    # One-time customer
    # --------------------------------------------------------

    history["is_one_time_customer"] = (
        history["total_bookings"] == 1
    )

    # --------------------------------------------------------
    # Customer lifetime
    # --------------------------------------------------------

    history["customer_lifetime_days"] = (
        history["last_booking_date"]
        -
        history["first_booking_date"]
    ).dt.days

    return history



# OVERALL RETENTION SUMMARY


def retention_summary(
    customers,
    history
):

    total_customers = len(customers)

    active_customers = len(history)

    repeat_customers = int(
        history["is_repeat_customer"]
        .sum()
    )

    one_time_customers = int(
        history["is_one_time_customer"]
        .sum()
    )

    inactive_customers = (
        total_customers -
        active_customers
    )

    repeat_customer_rate = (
        repeat_customers /
        active_customers *
        100
        if active_customers > 0
        else 0
    )

    one_time_rate = (
        one_time_customers /
        active_customers *
        100
        if active_customers > 0
        else 0
    )

    average_bookings = (
        history["total_bookings"]
        .mean()
    )

    median_bookings = (
        history["total_bookings"]
        .median()
    )

    average_revenue = (
        history["total_revenue"]
        .mean()
    )

    return {
        "total_customers": total_customers,
        "active_customers": active_customers,
        "inactive_customers": inactive_customers,
        "repeat_customers": repeat_customers,
        "one_time_customers": one_time_customers,
        "repeat_customer_rate": repeat_customer_rate,
        "one_time_rate": one_time_rate,
        "average_bookings": average_bookings,
        "median_bookings": median_bookings,
        "average_customer_revenue": average_revenue
    }



# BOOKING FREQUENCY DISTRIBUTION


def booking_frequency_distribution(history):

    distribution = (
        history
        .groupby("total_bookings")
        .agg(
            customers=(
                "customer_id",
                "count"
            ),

            revenue=(
                "total_revenue",
                "sum"
            )
        )
        .reset_index()
        .sort_values(
            "total_bookings"
        )
    )

    return distribution



# MONTHLY NEW VS RETURNING CUSTOMERS


def monthly_customer_type(
    bookings
):

    data = bookings.copy()

    # --------------------------------------------------------
    # First booking for each customer
    # --------------------------------------------------------

    first_booking = (
        data
        .groupby("customer_id")[
            "booking_date"
        ]
        .min()
        .rename(
            "first_booking_date"
        )
    )

    data = data.merge(
        first_booking,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Booking month
    # --------------------------------------------------------

    data["booking_month"] = (
        data["booking_date"]
        .dt.to_period("M")
    )

    data["first_booking_month"] = (
        data["first_booking_date"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # New / Returning classification
    # --------------------------------------------------------

    data["customer_type"] = "Returning"

    data.loc[
        data["booking_month"]
        ==
        data["first_booking_month"],
        "customer_type"
    ] = "New"

    # --------------------------------------------------------
    # Monthly summary
    # --------------------------------------------------------

    monthly = (
        data
        .groupby(
            [
                "booking_month",
                "customer_type"
            ]
        )
        .agg(
            customers=(
                "customer_id",
                "nunique"
            ),

            bookings=(
                "booking_id",
                "count"
            ),

            revenue=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    monthly["booking_month"] = (
        monthly["booking_month"]
        .astype(str)
    )

    return monthly



# MONTHLY CUSTOMER RETENTION


def monthly_retention(
    bookings
):

    data = bookings.copy()

    data["booking_month"] = (
        data["booking_date"]
        .dt.to_period("M")
    )

    # --------------------------------------------------------
    # First month for every customer
    # --------------------------------------------------------

    first_month = (
        data
        .groupby("customer_id")[
            "booking_month"
        ]
        .min()
        .rename("cohort_month")
    )

    data = data.merge(
        first_month,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Cohort age
    # --------------------------------------------------------

    data["cohort_index"] = (
        (
            data["booking_month"]
            -
            data["cohort_month"]
        )
        .apply(lambda x: x.n)
    )

    # --------------------------------------------------------
    # Number of customers in each cohort/month
    # --------------------------------------------------------

    retention = (
        data
        .groupby(
            [
                "cohort_month",
                "cohort_index"
            ]
        )["customer_id"]
        .nunique()
        .reset_index(
            name="active_customers"
        )
    )

    # --------------------------------------------------------
    # Cohort size
    # --------------------------------------------------------

    cohort_sizes = (
        retention[
            retention["cohort_index"] == 0
        ][
            [
                "cohort_month",
                "active_customers"
            ]
        ]
        .rename(
            columns={
                "active_customers":
                    "cohort_size"
            }
        )
    )

    retention = retention.merge(
        cohort_sizes,
        on="cohort_month",
        how="left"
    )

    # --------------------------------------------------------
    # Retention percentage
    # --------------------------------------------------------

    retention["retention_rate"] = (
        retention["active_customers"]
        /
        retention["cohort_size"]
        *
        100
    )

    retention["cohort_month"] = (
        retention["cohort_month"]
        .astype(str)
    )

    return retention



# REPEAT REVENUE ANALYSIS


def repeat_revenue_analysis(
    bookings
):

    data = bookings.copy()

    # --------------------------------------------------------
    # Determine first booking
    # --------------------------------------------------------

    first_booking = (
        data
        .groupby("customer_id")[
            "booking_date"
        ]
        .min()
        .rename(
            "first_booking_date"
        )
    )

    data = data.merge(
        first_booking,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # New vs repeat booking
    # --------------------------------------------------------

    data["booking_type"] = "Repeat"

    data.loc[
        data["booking_date"]
        ==
        data["first_booking_date"],
        "booking_type"
    ] = "First Booking"

    # --------------------------------------------------------
    # Revenue summary
    # --------------------------------------------------------

    result = (
        data
        .groupby("booking_type")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            customers=(
                "customer_id",
                "nunique"
            ),

            tickets=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    total_revenue = (
        result["revenue"]
        .sum()
    )

    result["revenue_percentage"] = (
        result["revenue"]
        /
        total_revenue
        *
        100
    )

    return result



# CUSTOMER RETENTION INSIGHTS


def generate_insights(
    summary,
    frequency,
    monthly_type,
    revenue
):

    print("\n")
    
    print("CUSTOMER RETENTION BUSINESS INSIGHTS")
    

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n1. OVERALL RETENTION")

    print(
        f"Active Customers       : "
        f"{summary['active_customers']:,}"
    )

    print(
        f"Repeat Customers       : "
        f"{summary['repeat_customers']:,}"
    )

    print(
        f"One-Time Customers     : "
        f"{summary['one_time_customers']:,}"
    )

    print(
        f"Repeat Customer Rate   : "
        f"{summary['repeat_customer_rate']:.2f}%"
    )

    print(
        f"One-Time Customer Rate : "
        f"{summary['one_time_rate']:.2f}%"
    )

    print(
        f"Average Bookings       : "
        f"{summary['average_bookings']:.2f}"
    )

    # --------------------------------------------------------
    # Retention interpretation
    # --------------------------------------------------------

    rate = summary[
        "repeat_customer_rate"
    ]

    print("\n2. RETENTION INTERPRETATION")

    if rate >= 60:

        print(
            "Customer retention is strong. "
            "A large proportion of active customers "
            "return for additional bookings."
        )

    elif rate >= 35:

        print(
            "Customer retention is moderate. "
            "There is an opportunity to convert "
            "more one-time customers into repeat customers."
        )

    else:

        print(
            "Customer retention is relatively low. "
            "The business should prioritize "
            "customer re-engagement and repeat booking strategies."
        )

    # --------------------------------------------------------
    # Most common frequency
    # --------------------------------------------------------

    if not frequency.empty:

        most_common = (
            frequency
            .sort_values(
                "customers",
                ascending=False
            )
            .iloc[0]
        )

        print("\n3. BOOKING FREQUENCY")

        print(
            f"Most Common Booking Frequency: "
            f"{int(most_common['total_bookings'])} booking(s)"
        )

        print(
            f"Customers: "
            f"{int(most_common['customers']):,}"
        )

    # --------------------------------------------------------
    # Revenue from repeat bookings
    # --------------------------------------------------------

    repeat_row = revenue[
        revenue["booking_type"]
        == "Repeat"
    ]

    if not repeat_row.empty:

        repeat_revenue_share = (
            repeat_row[
                "revenue_percentage"
            ].iloc[0]
        )

        print("\n4. REPEAT CUSTOMER REVENUE")

        print(
            f"Revenue from Repeat Bookings: "
            f"{repeat_revenue_share:.2f}%"
        )

    # --------------------------------------------------------
    # Business recommendations
    # --------------------------------------------------------

    print("\n5. BUSINESS RECOMMENDATIONS")

    if rate < 35:

        print(
            "• Create stronger second-booking incentives."
        )

        print(
            "• Introduce loyalty/reward programs."
        )

        print(
            "• Send personalized re-engagement campaigns."
        )

        print(
            "• Investigate why customers do not return."
        )

    elif rate < 60:

        print(
            "• Improve loyalty program adoption."
        )

        print(
            "• Target one-time customers with offers."
        )

        print(
            "• Personalize route recommendations."
        )

        print(
            "• Encourage customers to book again."
        )

    else:

        print(
            "• Protect the existing loyal customer base."
        )

        print(
            "• Reward high-frequency customers."
        )

        print(
            "• Introduce premium loyalty benefits."
        )



# SAVE RESULTS


def save_results(
    history,
    frequency,
    monthly_type,
    retention,
    revenue
):

    history_file = os.path.join(
        OUTPUT_DIR,
        "customer_booking_history.csv"
    )

    frequency_file = os.path.join(
        OUTPUT_DIR,
        "booking_frequency_distribution.csv"
    )

    monthly_file = os.path.join(
        OUTPUT_DIR,
        "monthly_new_vs_returning.csv"
    )

    retention_file = os.path.join(
        OUTPUT_DIR,
        "cohort_retention.csv"
    )

    revenue_file = os.path.join(
        OUTPUT_DIR,
        "first_vs_repeat_revenue.csv"
    )

    history.to_csv(
        history_file,
        index=False
    )

    frequency.to_csv(
        frequency_file,
        index=False
    )

    monthly_type.to_csv(
        monthly_file,
        index=False
    )

    retention.to_csv(
        retention_file,
        index=False
    )

    revenue.to_csv(
        revenue_file,
        index=False
    )

    print("\n")
    
    print("FILES SAVED")
    

    print(
        history_file
    )

    print(
        frequency_file
    )

    print(
        monthly_file
    )

    print(
        retention_file
    )

    print(
        revenue_file
    )



# MAIN


def main():

    
    print("STEP 9.3 — CUSTOMER RETENTION ANALYTICS")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    customers, bookings = load_data()

    print(
        f"\nCustomers loaded : "
        f"{len(customers):,}"
    )

    print(
        f"Bookings loaded  : "
        f"{len(bookings):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    bookings = prepare_bookings(
        bookings
    )

    print(
        f"Confirmed bookings: "
        f"{len(bookings):,}"
    )

    # --------------------------------------------------------
    # Customer history
    # --------------------------------------------------------

    print(
        "\nCalculating customer booking history..."
    )

    history = customer_booking_history(
        bookings
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary = retention_summary(
        customers,
        history
    )

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    frequency = (
        booking_frequency_distribution(
            history
        )
    )

    # --------------------------------------------------------
    # Monthly new vs returning
    # --------------------------------------------------------

    monthly_type = (
        monthly_customer_type(
            bookings
        )
    )

    # --------------------------------------------------------
    # Cohort retention
    # --------------------------------------------------------

    retention = (
        monthly_retention(
            bookings
        )
    )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    revenue = (
        repeat_revenue_analysis(
            bookings
        )
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    
    print("CUSTOMER RETENTION SUMMARY")
    

    print(
        f"Total Customers          : "
        f"{summary['total_customers']:,}"
    )

    print(
        f"Active Customers         : "
        f"{summary['active_customers']:,}"
    )

    print(
        f"Inactive Customers       : "
        f"{summary['inactive_customers']:,}"
    )

    print(
        f"Repeat Customers         : "
        f"{summary['repeat_customers']:,}"
    )

    print(
        f"One-Time Customers       : "
        f"{summary['one_time_customers']:,}"
    )

    print(
        f"Repeat Customer Rate     : "
        f"{summary['repeat_customer_rate']:.2f}%"
    )

    print(
        f"One-Time Customer Rate   : "
        f"{summary['one_time_rate']:.2f}%"
    )

    print(
        f"Average Bookings/Customer: "
        f"{summary['average_bookings']:.2f}"
    )

    print(
        f"Median Bookings/Customer : "
        f"{summary['median_bookings']:.2f}"
    )

    print(
        f"Average Customer Revenue : "
        f"₹{summary['average_customer_revenue']:,.2f}"
    )

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    print("\n")
    
    print("BOOKING FREQUENCY DISTRIBUTION")
    

    print(
        frequency.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    print("\n")
    
    print("MONTHLY NEW VS RETURNING CUSTOMERS")
    

    print(
        monthly_type.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    print("\n")
    
    print("FIRST VS REPEAT BOOKING REVENUE")
    

    print(
        revenue.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Cohort
    # --------------------------------------------------------

    print("\n")
    
    print("COHORT RETENTION")
    

    print(
        retention.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_insights(
        summary,
        frequency,
        monthly_type,
        revenue
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        history,
        frequency,
        monthly_type,
        retention,
        revenue
    )

    print("\n")
    
    print("STEP 9.3 COMPLETED")
    


if __name__ == "__main__":
    main()