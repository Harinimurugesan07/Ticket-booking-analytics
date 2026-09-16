import os
import pandas as pd
import matplotlib.pyplot as plt



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

CHART_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "charts"
)

INSIGHT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "insights"
)

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(INSIGHT_DIR, exist_ok=True)



# LOAD DATA


def load_data():

    customers = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "customers_clean.csv"
        )
    )

    operators = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "operators_clean.csv"
        )
    )

    routes = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "routes_clean.csv"
        )
    )

    trips = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "trips_clean.csv"
        )
    )

    bookings = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "bookings_clean.csv"
        )
    )

    # Convert dates

    bookings["booking_date"] = pd.to_datetime(
        bookings["booking_date"]
    )

    bookings["travel_date"] = pd.to_datetime(
        bookings["travel_date"]
    )

    trips["departure_datetime"] = pd.to_datetime(
        trips["departure_datetime"]
    )

    trips["arrival_datetime"] = pd.to_datetime(
        trips["arrival_datetime"]
    )

    return (
        customers,
        operators,
        routes,
        trips,
        bookings
    )



# PREPARE COMBINED DATA


def prepare_booking_data(
    bookings,
    trips,
    routes,
    operators,
    customers
):

    data = bookings.merge(
        trips[
            [
                "trip_id",
                "route_id",
                "operator_id",
                "vehicle_type",
                "total_seats"
            ]
        ],
        on="trip_id",
        how="left"
    )

    data = data.merge(
        routes[
            [
                "route_id",
                "route_name",
                "source",
                "destination",
                "distance_km"
            ]
        ],
        on="route_id",
        how="left"
    )

    data = data.merge(
        operators[
            [
                "operator_id",
                "operator_name"
            ]
        ],
        on="operator_id",
        how="left"
    )

    data = data.merge(
        customers[
            [
                "customer_id",
                "customer_name",
                "gender",
                "age",
                "city"
            ]
        ],
        on="customer_id",
        how="left"
    )

    return data



# CHART 1 — BOOKING STATUS


def booking_status_chart(bookings):

    counts = (
        bookings["booking_status"]
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Booking Status Distribution"
    )

    plt.xlabel(
        "Booking Status"
    )

    plt.ylabel(
        "Number of Bookings"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "01_booking_status.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 2 — PAYMENT METHOD


def payment_method_chart(bookings):

    counts = (
        bookings["payment_method"]
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Bookings by Payment Method"
    )

    plt.xlabel(
        "Payment Method"
    )

    plt.ylabel(
        "Number of Bookings"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "02_payment_method.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 3 — VEHICLE TYPE BOOKINGS


def vehicle_booking_chart(data):

    result = (
        data
        .groupby("vehicle_type")
        .size()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(8, 5))

    result.plot(
        kind="bar"
    )

    plt.title(
        "Bookings by Vehicle Type"
    )

    plt.xlabel(
        "Vehicle Type"
    )

    plt.ylabel(
        "Number of Bookings"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "03_vehicle_bookings.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 4 — REVENUE BY VEHICLE


def vehicle_revenue_chart(data):

    result = (
        data
        .groupby("vehicle_type")[
            "total_amount"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(figsize=(8, 5))

    result.plot(
        kind="bar"
    )

    plt.title(
        "Revenue by Vehicle Type"
    )

    plt.xlabel(
        "Vehicle Type"
    )

    plt.ylabel(
        "Revenue"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "04_vehicle_revenue.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 5 — TOP ROUTES


def top_routes_chart(data):

    result = (
        data
        .groupby("route_name")
        .size()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    result.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Routes by Bookings"
    )

    plt.xlabel(
        "Number of Bookings"
    )

    plt.ylabel(
        "Route"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "05_top_routes.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 6 — TOP ROUTES BY REVENUE


def top_revenue_routes_chart(data):

    result = (
        data
        .groupby("route_name")[
            "total_amount"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    plt.figure(figsize=(10, 6))

    result.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Routes by Revenue"
    )

    plt.xlabel(
        "Revenue"
    )

    plt.ylabel(
        "Route"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "06_top_revenue_routes.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 7 — DAILY BOOKINGS


def daily_bookings_chart(bookings):

    result = (
        bookings
        .groupby("booking_date")
        .size()
    )

    plt.figure(figsize=(12, 5))

    result.plot()

    plt.title(
        "Daily Booking Trend"
    )

    plt.xlabel(
        "Booking Date"
    )

    plt.ylabel(
        "Number of Bookings"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "07_daily_bookings.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 8 — DAILY REVENUE


def daily_revenue_chart(bookings):

    result = (
        bookings
        .groupby("booking_date")[
            "total_amount"
        ]
        .sum()
    )

    plt.figure(figsize=(12, 5))

    result.plot()

    plt.title(
        "Daily Revenue Trend"
    )

    plt.xlabel(
        "Booking Date"
    )

    plt.ylabel(
        "Revenue"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "08_daily_revenue.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 9 — MONTHLY REVENUE


def monthly_revenue_chart(bookings):

    data = bookings.copy()

    data["month"] = (
        data["booking_date"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        data
        .groupby("month")[
            "total_amount"
        ]
        .sum()
    )

    plt.figure(figsize=(12, 5))

    result.plot(
        kind="bar"
    )

    plt.title(
        "Monthly Revenue"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Revenue"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "09_monthly_revenue.png"
        ),
        dpi=150
    )

    plt.close()



# CHART 10 — MONTHLY BOOKINGS


def monthly_bookings_chart(bookings):

    data = bookings.copy()

    data["month"] = (
        data["booking_date"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        data
        .groupby("month")
        .size()
    )

    plt.figure(figsize=(12, 5))

    result.plot(
        kind="bar"
    )

    plt.title(
        "Monthly Booking Volume"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Bookings"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "10_monthly_bookings.png"
        ),
        dpi=150
    )

    plt.close()



# BUSINESS INSIGHTS


def generate_insights(
    bookings,
    data
):

    total_bookings = len(bookings)

    total_revenue = (
        bookings["total_amount"]
        .sum()
    )

    average_booking_value = (
        bookings["total_amount"]
        .mean()
    )

    cancelled = (
        bookings["booking_status"]
        == "CANCELLED"
    ).sum()

    cancellation_rate = (
        cancelled /
        total_bookings *
        100
    )

    # Top vehicle

    vehicle_bookings = (
        data
        .groupby("vehicle_type")
        .size()
        .sort_values(
            ascending=False
        )
    )

    top_vehicle = (
        vehicle_bookings
        .index[0]
    )

    top_vehicle_count = (
        vehicle_bookings
        .iloc[0]
    )

    # Top route

    route_bookings = (
        data
        .groupby("route_name")
        .size()
        .sort_values(
            ascending=False
        )
    )

    top_route = (
        route_bookings
        .index[0]
    )

    top_route_count = (
        route_bookings
        .iloc[0]
    )

    # Top payment method

    payment_methods = (
        bookings[
            "payment_method"
        ]
        .value_counts()
    )

    top_payment = (
        payment_methods
        .index[0]
    )

    top_payment_count = (
        payment_methods
        .iloc[0]
    )

    # Top operator

    operator_revenue = (
        data
        .groupby("operator_name")[
            "total_amount"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_operator = (
        operator_revenue
        .index[0]
    )

    top_operator_revenue = (
        operator_revenue
        .iloc[0]
    )

    # --------------------------------------------------------
    # CREATE REPORT
    # --------------------------------------------------------

    report = f"""
============================================================
TICKET BOOKING ANALYTICS
BUSINESS INSIGHTS REPORT
============================================================

1. OVERALL PERFORMANCE
------------------------------------------------------------

Total Bookings:
{total_bookings:,}

Total Revenue:
₹{total_revenue:,.2f}

Average Booking Value:
₹{average_booking_value:,.2f}

Cancellation Rate:
{cancellation_rate:.2f}%


2. VEHICLE PERFORMANCE
------------------------------------------------------------

Most Popular Vehicle Type:
{top_vehicle}

Bookings:
{top_vehicle_count:,}


3. ROUTE PERFORMANCE
------------------------------------------------------------

Top Route by Booking Volume:
{top_route}

Bookings:
{top_route_count:,}


4. PAYMENT BEHAVIOR
------------------------------------------------------------

Most Used Payment Method:
{top_payment}

Bookings:
{top_payment_count:,}


5. OPERATOR PERFORMANCE
------------------------------------------------------------

Top Operator by Revenue:
{top_operator}

Revenue:
₹{top_operator_revenue:,.2f}


6. BUSINESS INTERPRETATION
------------------------------------------------------------

• The most popular vehicle type indicates where
  customer demand is strongest.

• The highest-booked route represents the route
  with the strongest customer demand.

• The most frequently used payment method shows
  the preferred payment channel of customers.

• The top operator contributes the highest revenue
  among available operators.

• Cancellation rate should be monitored because
  high cancellations can reduce realized revenue.

============================================================
END OF REPORT
============================================================
"""

    report_path = os.path.join(
        INSIGHT_DIR,
        "business_insights.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(report)



# MAIN


def main():

    
    print("BUSINESS INSIGHTS & VISUALIZATION")
    

    (
        customers,
        operators,
        routes,
        trips,
        bookings
    ) = load_data()

    print("\nPreparing combined dataset...")

    data = prepare_booking_data(
        bookings,
        trips,
        routes,
        operators,
        customers
    )

    print("Generating charts...\n")

    booking_status_chart(
        bookings
    )

    payment_method_chart(
        bookings
    )

    vehicle_booking_chart(
        data
    )

    vehicle_revenue_chart(
        data
    )

    top_routes_chart(
        data
    )

    top_revenue_routes_chart(
        data
    )

    daily_bookings_chart(
        bookings
    )

    daily_revenue_chart(
        bookings
    )

    monthly_revenue_chart(
        bookings
    )

    monthly_bookings_chart(
        bookings
    )

    print("All charts generated successfully.")

    generate_insights(
        bookings,
        data
    )

    print("\n")
    
    print("STEP 8 COMPLETED")
    

    print(
        f"\nCharts location:\n{CHART_DIR}"
    )

    print(
        f"\nInsights location:\n{INSIGHT_DIR}"
    )


if __name__ == "__main__":
    main()