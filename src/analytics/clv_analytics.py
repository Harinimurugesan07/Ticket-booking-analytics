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

    bookings = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "bookings_clean.csv"
        )
    )

    bookings["booking_date"] = pd.to_datetime(
        bookings["booking_date"],
        errors="coerce"
    )

    bookings["travel_date"] = pd.to_datetime(
        bookings["travel_date"],
        errors="coerce"
    )

    return bookings



# PREPARE CONFIRMED BOOKINGS


def prepare_data(bookings):

    data = bookings.copy()

    # --------------------------------------------------------
    # Only confirmed bookings generate realized revenue
    # --------------------------------------------------------

    data = data[
        data["booking_status"]
        .astype(str)
        .str.upper()
        .eq("CONFIRMED")
    ].copy()

    return data



# CUSTOMER METRICS


def calculate_customer_metrics(data):

    customer = (
        data
        .groupby("customer_id")
        .agg(
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
            ),

            first_booking_date=(
                "booking_date",
                "min"
            ),

            last_booking_date=(
                "booking_date",
                "max"
            ),

            average_booking_value=(
                "total_amount",
                "mean"
            ),

            average_ticket_price=(
                "ticket_price",
                "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Customer lifetime in days
    # --------------------------------------------------------

    customer["lifetime_days"] = (
        customer["last_booking_date"]
        -
        customer["first_booking_date"]
    ).dt.days

    customer["lifetime_days"] = (
        customer["lifetime_days"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Customer lifetime in months
    # --------------------------------------------------------

    customer["lifetime_months"] = (
        customer["lifetime_days"]
        /
        30
    )

    # --------------------------------------------------------
    # Purchase frequency
    # --------------------------------------------------------

    customer["purchase_frequency"] = (
        customer["total_bookings"]
        /
        customer["lifetime_months"].clip(
            lower=1
        )
    )

    return customer



# CLV CALCULATION


def calculate_clv(customer):

    customer = customer.copy()

    # --------------------------------------------------------
    # Historical CLV
    #
    # Since this project does not yet have actual profit
    # margin or future churn predictions, we use historical
    # customer revenue as the base CLV measure.
    # --------------------------------------------------------

    customer["historical_clv"] = (
        customer["total_revenue"]
    )

    # --------------------------------------------------------
    # Revenue per month
    # --------------------------------------------------------

    customer["monthly_revenue"] = (
        customer["total_revenue"]
        /
        customer["lifetime_months"].clip(
            lower=1
        )
    )

    # --------------------------------------------------------
    # Simple projected 12-month CLV
    #
    # This is a revenue-based projection, not profit.
    # --------------------------------------------------------

    customer["projected_12_month_clv"] = (
        customer["monthly_revenue"]
        *
        12
    )

    return customer



# CUSTOMER VALUE SEGMENTS


def create_value_segments(customer):

    customer = customer.copy()

    # --------------------------------------------------------
    # Use percentile-based segmentation.
    #
    # This is better than arbitrary fixed rupee thresholds
    # because the thresholds adapt to your dataset.
    # --------------------------------------------------------

    q25 = customer["historical_clv"].quantile(0.25)
    q50 = customer["historical_clv"].quantile(0.50)
    q75 = customer["historical_clv"].quantile(0.75)

    def segment(value):

        if value >= q75:

            return "VIP"

        elif value >= q50:

            return "High Value"

        elif value >= q25:

            return "Medium Value"

        else:

            return "Low Value"

    customer["customer_value_segment"] = (
        customer["historical_clv"]
        .apply(segment)
    )

    return customer



# CUSTOMER RANKING


def rank_customers(customer):

    customer = customer.copy()

    customer["revenue_rank"] = (
        customer["total_revenue"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    customer["clv_rank"] = (
        customer["historical_clv"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    return customer.sort_values(
        "historical_clv",
        ascending=False
    )



# CLV SEGMENT SUMMARY


def segment_summary(customer):

    result = (
        customer
        .groupby(
            "customer_value_segment"
        )
        .agg(
            customers=(
                "customer_id",
                "count"
            ),

            total_revenue=(
                "total_revenue",
                "sum"
            ),

            average_revenue=(
                "total_revenue",
                "mean"
            ),

            average_bookings=(
                "total_bookings",
                "mean"
            ),

            average_tickets=(
                "total_tickets",
                "mean"
            ),

            average_clv=(
                "historical_clv",
                "mean"
            ),

            average_projected_clv=(
                "projected_12_month_clv",
                "mean"
            )
        )
        .reset_index()
    )

    result["revenue_share"] = (
        result["total_revenue"]
        /
        result["total_revenue"].sum()
        *
        100
    )

    return result



# CUSTOMER REPEAT ANALYSIS


def repeat_customer_analysis(customer):

    customer = customer.copy()

    repeat = customer.copy()

    repeat["customer_type"] = (
        repeat["total_bookings"]
        .apply(
            lambda x:
            "Repeat Customer"
            if x > 1
            else "One-Time Customer"
        )
    )

    result = (
        repeat
        .groupby("customer_type")
        .agg(
            customers=(
                "customer_id",
                "count"
            ),

            total_revenue=(
                "total_revenue",
                "sum"
            ),

            average_revenue=(
                "total_revenue",
                "mean"
            ),

            average_bookings=(
                "total_bookings",
                "mean"
            ),

            average_clv=(
                "historical_clv",
                "mean"
            )
        )
        .reset_index()
    )

    result["customer_share"] = (
        result["customers"]
        /
        result["customers"].sum()
        *
        100
    )

    result["revenue_share"] = (
        result["total_revenue"]
        /
        result["total_revenue"].sum()
        *
        100
    )

    return result



# REVENUE CONCENTRATION


def revenue_concentration(customer):

    customer = customer.sort_values(
        "total_revenue",
        ascending=False
    ).copy()

    total_revenue = (
        customer["total_revenue"]
        .sum()
    )

    customer["cumulative_revenue"] = (
        customer["total_revenue"]
        .cumsum()
    )

    customer["cumulative_revenue_percentage"] = (
        customer["cumulative_revenue"]
        /
        total_revenue
        *
        100
    )

    # --------------------------------------------------------
    # Top 10% customer revenue
    # --------------------------------------------------------

    top_10_count = max(
        1,
        int(len(customer) * 0.10)
    )

    top_10_revenue = (
        customer
        .head(top_10_count)["total_revenue"]
        .sum()
    )

    top_10_percentage = (
        top_10_revenue
        /
        total_revenue
        *
        100
    )

    return customer, top_10_percentage



# OVERALL CLV SUMMARY


def overall_summary(customer):

    total_customers = len(customer)

    total_revenue = (
        customer["total_revenue"]
        .sum()
    )

    average_clv = (
        customer["historical_clv"]
        .mean()
    )

    median_clv = (
        customer["historical_clv"]
        .median()
    )

    maximum_clv = (
        customer["historical_clv"]
        .max()
    )

    average_bookings = (
        customer["total_bookings"]
        .mean()
    )

    repeat_customers = (
        customer["total_bookings"] > 1
    ).sum()

    repeat_rate = (
        repeat_customers
        /
        total_customers
        *
        100
    )

    return {
        "total_customers":
            total_customers,

        "total_revenue":
            total_revenue,

        "average_clv":
            average_clv,

        "median_clv":
            median_clv,

        "maximum_clv":
            maximum_clv,

        "average_bookings":
            average_bookings,

        "repeat_customers":
            repeat_customers,

        "repeat_rate":
            repeat_rate
    }



# BUSINESS INSIGHTS


def generate_insights(
    summary,
    customer,
    segments,
    repeat,
    top_10_percentage
):

    print("\n")
    
    print("CUSTOMER LIFETIME VALUE BUSINESS INSIGHTS")
    

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n1. OVERALL CUSTOMER VALUE")

    print(
        f"Total Customers       : "
        f"{summary['total_customers']:,}"
    )

    print(
        f"Total Customer Revenue: "
        f"₹{summary['total_revenue']:,.2f}"
    )

    print(
        f"Average CLV           : "
        f"₹{summary['average_clv']:,.2f}"
    )

    print(
        f"Median CLV            : "
        f"₹{summary['median_clv']:,.2f}"
    )

    print(
        f"Maximum CLV           : "
        f"₹{summary['maximum_clv']:,.2f}"
    )

    print(
        f"Average Bookings      : "
        f"{summary['average_bookings']:.2f}"
    )

    print(
        f"Repeat Customers      : "
        f"{summary['repeat_customers']:,}"
    )

    print(
        f"Repeat Customer Rate  : "
        f"{summary['repeat_rate']:.2f}%"
    )

    # --------------------------------------------------------
    # Top customers
    # --------------------------------------------------------

    print("\n2. TOP 10 CUSTOMERS BY CLV")
    print("-" * 80)

    print(
        customer[
            [
                "customer_id",
                "total_bookings",
                "total_tickets",
                "total_revenue",
                "historical_clv",
                "projected_12_month_clv",
                "customer_value_segment"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Segment summary
    # --------------------------------------------------------

    print("\n3. CUSTOMER VALUE SEGMENTS")
    print("-" * 80)

    print(
        segments.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Repeat customers
    # --------------------------------------------------------

    print("\n4. REPEAT CUSTOMER VALUE")
    print("-" * 80)

    print(
        repeat.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Revenue concentration
    # --------------------------------------------------------

    print("\n5. REVENUE CONCENTRATION")
    print("-" * 80)

    print(
        f"Top 10% customers generate "
        f"{top_10_percentage:.2f}% "
        f"of total customer revenue."
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    print("\n6. BUSINESS RECOMMENDATIONS")
    print("-" * 80)

    print(
        "• Protect VIP customers with loyalty and "
        "retention strategies."
    )

    print(
        "• Encourage High Value customers to increase "
        "booking frequency."
    )

    print(
        "• Use targeted offers to move Medium Value "
        "customers toward higher-value segments."
    )

    print(
        "• Analyze Low Value customers before spending "
        "large amounts on acquisition incentives."
    )

    if summary["repeat_rate"] < 30:

        print(
            "• Repeat customer rate is relatively low; "
            "retention should be a priority."
        )

    elif summary["repeat_rate"] < 60:

        print(
            "• Repeat customer behavior is moderate; "
            "loyalty programs may increase retention."
        )

    else:

        print(
            "• Repeat customer behavior is strong; "
            "focus on protecting high-value customers."
        )

    if top_10_percentage > 40:

        print(
            "• Revenue is highly concentrated among top "
            "customers; protect these customers carefully."
        )



# SAVE RESULTS


def save_results(
    customer,
    segments,
    repeat,
    concentration
):

    results = {

        "customer_clv.csv":
            customer,

        "clv_segments.csv":
            segments,

        "repeat_customer_clv.csv":
            repeat,

        "customer_revenue_concentration.csv":
            concentration
    }

    print("\n")
    
    print("FILES SAVED")
    

    for filename, dataframe in results.items():

        filepath = os.path.join(
            OUTPUT_DIR,
            filename
        )

        dataframe.to_csv(
            filepath,
            index=False
        )

        print(filepath)



# MAIN


def main():

    
    print("STEP 9.7 — CUSTOMER LIFETIME VALUE ANALYTICS")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    bookings = load_data()

    print(
        f"\nBookings loaded : "
        f"{len(bookings):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        bookings
    )

    print(
        f"Confirmed bookings: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Customer metrics
    # --------------------------------------------------------

    customer = calculate_customer_metrics(
        data
    )

    print(
        f"Customers analyzed: "
        f"{len(customer):,}"
    )

    # --------------------------------------------------------
    # CLV
    # --------------------------------------------------------

    customer = calculate_clv(
        customer
    )

    # --------------------------------------------------------
    # Segments
    # --------------------------------------------------------

    customer = create_value_segments(
        customer
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    customer = rank_customers(
        customer
    )

    # --------------------------------------------------------
    # Segment summary
    # --------------------------------------------------------

    segments = segment_summary(
        customer
    )

    # --------------------------------------------------------
    # Repeat analysis
    # --------------------------------------------------------

    repeat = repeat_customer_analysis(
        customer
    )

    # --------------------------------------------------------
    # Revenue concentration
    # --------------------------------------------------------

    concentration, top_10_percentage = (
        revenue_concentration(
            customer
        )
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary = overall_summary(
        customer
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_insights(
        summary,
        customer,
        segments,
        repeat,
        top_10_percentage
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        customer,
        segments,
        repeat,
        concentration
    )

    print("\n")
    
    print("STEP 9.7 COMPLETED")
    


if __name__ == "__main__":
    main()