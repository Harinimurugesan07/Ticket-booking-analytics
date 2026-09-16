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



# CUSTOMER ANALYTICS


def customer_analysis(
    customers,
    bookings
):

    # --------------------------------------------------------
    # Aggregate booking behavior per customer
    # --------------------------------------------------------

    customer_metrics = (
        bookings
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

            total_spend=(
                "total_amount",
                "sum"
            ),

            average_booking_value=(
                "total_amount",
                "mean"
            ),

            first_booking_date=(
                "booking_date",
                "min"
            ),

            last_booking_date=(
                "booking_date",
                "max"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Join customer information
    # --------------------------------------------------------

    customer_metrics = customer_metrics.merge(
        customers[
            [
                "customer_id",
                "customer_name",
                "gender",
                "age",
                "city",
                "registration_date"
            ]
        ],
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------------
    # Calculate customer lifetime days
    # --------------------------------------------------------

    customer_metrics["customer_lifetime_days"] = (
        customer_metrics["last_booking_date"]
        -
        customer_metrics["first_booking_date"]
    ).dt.days

    # --------------------------------------------------------
    # Repeat customer flag
    # --------------------------------------------------------

    customer_metrics["is_repeat_customer"] = (
        customer_metrics["total_bookings"] > 1
    )

    return customer_metrics



# CUSTOMER SUMMARY


def customer_summary(
    customers,
    customer_metrics
):

    total_customers = len(customers)

    active_customers = len(
        customer_metrics
    )

    repeat_customers = (
        customer_metrics[
            "is_repeat_customer"
        ].sum()
    )

    one_time_customers = (
        active_customers -
        repeat_customers
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

    average_customer_spend = (
        customer_metrics[
            "total_spend"
        ].mean()
    )

    average_bookings_per_customer = (
        customer_metrics[
            "total_bookings"
        ].mean()
    )

    return {
        "total_customers": total_customers,
        "active_customers": active_customers,
        "inactive_customers": inactive_customers,
        "repeat_customers": repeat_customers,
        "one_time_customers": one_time_customers,
        "repeat_customer_rate": repeat_customer_rate,
        "average_customer_spend": average_customer_spend,
        "average_bookings_per_customer":
            average_bookings_per_customer
    }



# MAIN


def main():

    
    print("STEP 9 — ADVANCED BUSINESS ANALYTICS")
    

    customers, bookings = load_data()

    # Customer metrics

    customer_metrics = customer_analysis(
        customers,
        bookings
    )

    # Summary

    summary = customer_summary(
        customers,
        customer_metrics
    )

    print("\n")
    
    print("CUSTOMER SUMMARY")
    

    print(
        f"Total Customers              : "
        f"{summary['total_customers']:,}"
    )

    print(
        f"Active Customers             : "
        f"{summary['active_customers']:,}"
    )

    print(
        f"Inactive Customers           : "
        f"{summary['inactive_customers']:,}"
    )

    print(
        f"Repeat Customers             : "
        f"{summary['repeat_customers']:,}"
    )

    print(
        f"One-Time Customers           : "
        f"{summary['one_time_customers']:,}"
    )

    print(
        f"Repeat Customer Rate         : "
        f"{summary['repeat_customer_rate']:.2f}%"
    )

    print(
        f"Average Customer Spend       : "
        f"₹{summary['average_customer_spend']:,.2f}"
    )

    print(
        f"Average Bookings/Customer    : "
        f"{summary['average_bookings_per_customer']:.2f}"
    )

    # --------------------------------------------------------
    # TOP CUSTOMERS
    # --------------------------------------------------------

    print("\n")
    
    print("TOP 10 CUSTOMERS BY SPENDING")
    

    top_customers = (
        customer_metrics
        .sort_values(
            "total_spend",
            ascending=False
        )
        .head(10)
    )

    print(
        top_customers[
            [
                "customer_id",
                "customer_name",
                "city",
                "total_bookings",
                "total_tickets",
                "total_spend",
                "average_booking_value"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # TOP CUSTOMERS BY BOOKING FREQUENCY
    # --------------------------------------------------------

    print("\n")
    
    print("TOP 10 CUSTOMERS BY BOOKING FREQUENCY")
    

    frequent_customers = (
        customer_metrics
        .sort_values(
            "total_bookings",
            ascending=False
        )
        .head(10)
    )

    print(
        frequent_customers[
            [
                "customer_id",
                "customer_name",
                "city",
                "total_bookings",
                "total_tickets",
                "total_spend"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE CUSTOMER ANALYTICS
    # --------------------------------------------------------

    output_dir = os.path.join(
        BASE_DIR,
        "reports",
        "analytics"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "customer_analytics.csv"
    )

    customer_metrics.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print(
        f"Customer analytics saved to:\n"
        f"{output_file}"
    )

    print("\n")
    
    print("CUSTOMER ANALYTICS COMPLETED")
    


if __name__ == "__main__":
    main()