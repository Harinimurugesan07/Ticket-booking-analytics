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

    trips = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "trips_clean.csv"
        )
    )

    routes = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "routes_clean.csv"
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

    trips["departure_datetime"] = pd.to_datetime(
        trips["departure_datetime"],
        errors="coerce"
    )

    return bookings, trips, routes



# PREPARE DATA


def prepare_data(
    bookings,
    trips,
    routes
):

    data = bookings.copy()

    # --------------------------------------------------------
    # Join trip information
    # --------------------------------------------------------

    trip_columns = [
        "trip_id",
        "route_id",
        "vehicle_type",
        "departure_datetime"
    ]

    trip_columns = [
        column
        for column in trip_columns
        if column in trips.columns
    ]

    data = data.merge(
        trips[trip_columns],
        on="trip_id",
        how="left"
    )

    # --------------------------------------------------------
    # Join route information
    # --------------------------------------------------------

    route_columns = [
        "route_id",
        "route_name",
        "source",
        "destination"
    ]

    route_columns = [
        column
        for column in route_columns
        if column in routes.columns
    ]

    data = data.merge(
        routes[route_columns],
        on="route_id",
        how="left"
    )

    # --------------------------------------------------------
    # Route label
    # --------------------------------------------------------

    data["route"] = (
        data["source"].astype(str)
        + " → "
        + data["destination"].astype(str)
    )

    # --------------------------------------------------------
    # Booking status normalized
    # --------------------------------------------------------

    data["booking_status"] = (
        data["booking_status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Cancellation flag
    # --------------------------------------------------------

    data["is_cancelled"] = (
        data["booking_status"]
        == "CANCELLED"
    )

    # --------------------------------------------------------
    # Confirmation flag
    # --------------------------------------------------------

    data["is_confirmed"] = (
        data["booking_status"]
        == "CONFIRMED"
    )

    # --------------------------------------------------------
    # Pending flag
    # --------------------------------------------------------

    data["is_pending"] = (
        data["booking_status"]
        == "PENDING"
    )

    # --------------------------------------------------------
    # Booking month
    # --------------------------------------------------------

    data["booking_month"] = (
        data["booking_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------------
    # Booking day of week
    # --------------------------------------------------------

    data["booking_day_of_week"] = (
        data["booking_date"]
        .dt.day_name()
    )

    # --------------------------------------------------------
    # Travel month
    # --------------------------------------------------------

    data["travel_month"] = (
        data["travel_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------------
    # Booking lead time
    #
    # Days between booking date and travel date
    # --------------------------------------------------------

    data["lead_time_days"] = (
        data["travel_date"]
        -
        data["booking_date"]
    ).dt.days

    return data



# OVERALL CANCELLATION SUMMARY


def overall_cancellation_summary(data):

    total_bookings = len(data)

    cancelled = (
        data["is_cancelled"]
        .sum()
    )

    confirmed = (
        data["is_confirmed"]
        .sum()
    )

    pending = (
        data["is_pending"]
        .sum()
    )

    cancellation_rate = (
        cancelled
        /
        total_bookings
        *
        100
        if total_bookings > 0
        else 0
    )

    # Revenue associated with cancelled bookings
    cancelled_revenue = (
        data.loc[
            data["is_cancelled"],
            "total_amount"
        ]
        .sum()
    )

    confirmed_revenue = (
        data.loc[
            data["is_confirmed"],
            "total_amount"
        ]
        .sum()
    )

    total_booking_value = (
        data["total_amount"]
        .sum()
    )

    cancelled_value_share = (
        cancelled_revenue
        /
        total_booking_value
        *
        100
        if total_booking_value > 0
        else 0
    )

    return {
        "total_bookings": total_bookings,
        "confirmed_bookings": confirmed,
        "cancelled_bookings": cancelled,
        "pending_bookings": pending,
        "cancellation_rate": cancellation_rate,
        "cancelled_revenue": cancelled_revenue,
        "confirmed_revenue": confirmed_revenue,
        "cancelled_value_share": cancelled_value_share
    }



# STATUS DISTRIBUTION


def status_distribution(data):

    result = (
        data
        .groupby("booking_status")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
                "seat_count",
                "sum"
            ),

            booking_value=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    result["booking_share"] = (
        result["bookings"]
        /
        result["bookings"].sum()
        *
        100
    )

    return result.sort_values(
        "bookings",
        ascending=False
    )



# CANCELLATION BY ROUTE


def route_cancellation(data):

    result = (
        data
        .groupby(
            [
                "route_id",
                "route",
                "source",
                "destination"
            ]
        )
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            ),

            total_booking_value=(
                "total_amount",
                "sum"
            ),

            cancelled_value=(
                "is_cancelled",
                lambda x:
                data.loc[
                    x.index,
                    "total_amount"
                ][x].sum()
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    result["cancelled_value_share"] = (
        result["cancelled_value"]
        /
        result["total_booking_value"]
        *
        100
    )

    return result.sort_values(
        "cancellation_rate",
        ascending=False
    )



# CANCELLATION BY PAYMENT METHOD


def payment_cancellation(data):

    result = (
        data
        .groupby("payment_method")
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            ),

            booking_value=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    return result.sort_values(
        "cancellation_rate",
        ascending=False
    )



# CANCELLATION BY VEHICLE TYPE


def vehicle_cancellation(data):

    result = (
        data
        .groupby("vehicle_type")
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            ),

            booking_value=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    return result.sort_values(
        "cancellation_rate",
        ascending=False
    )



# MONTHLY CANCELLATION


def monthly_cancellation(data):

    result = (
        data
        .groupby("booking_month")
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            ),

            booking_value=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    return result.sort_values(
        "booking_month"
    )



# DAY OF WEEK CANCELLATION


def weekday_cancellation(data):

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    result = (
        data
        .groupby("booking_day_of_week")
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    result["day_order"] = (
        result["booking_day_of_week"]
        .map(
            {
                day: i
                for i, day
                in enumerate(weekday_order)
            }
        )
    )

    return (
        result
        .sort_values("day_order")
        .drop(columns="day_order")
    )



# LEAD TIME ANALYSIS


def lead_time_analysis(data):

    data = data.copy()

    # --------------------------------------------------------
    # Create lead-time bands
    # --------------------------------------------------------

    data["lead_time_band"] = pd.cut(
        data["lead_time_days"],
        bins=[
            -1,
            1,
            3,
            7,
            14,
            30,
            float("inf")
        ],
        labels=[
            "0–1 days",
            "2–3 days",
            "4–7 days",
            "8–14 days",
            "15–30 days",
            "30+ days"
        ]
    )

    result = (
        data
        .groupby(
            "lead_time_band",
            observed=True
        )
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            ),

            booking_value=(
                "total_amount",
                "sum"
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    return result



# CUSTOMER CANCELLATION BEHAVIOR


def customer_cancellation(data):

    result = (
        data
        .groupby("customer_id")
        .agg(
            total_bookings=(
                "booking_id",
                "count"
            ),

            cancelled_bookings=(
                "is_cancelled",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            ),

            total_booking_value=(
                "total_amount",
                "sum"
            ),

            cancelled_value=(
                "is_cancelled",
                lambda x:
                data.loc[
                    x.index,
                    "total_amount"
                ][x].sum()
            )
        )
        .reset_index()
    )

    result["cancellation_rate"] = (
        result["cancelled_bookings"]
        /
        result["total_bookings"]
        *
        100
    )

    # --------------------------------------------------------
    # Frequent cancellers
    # --------------------------------------------------------

    result["frequent_canceller"] = (
        (
            result["total_bookings"] >= 3
        )
        &
        (
            result["cancellation_rate"] >= 50
        )
    )

    return result.sort_values(
        [
            "frequent_canceller",
            "cancellation_rate",
            "cancelled_bookings"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )



# HIGH CANCELLATION ROUTES


def high_cancellation_routes(route):

    return (
        route[
            route["total_bookings"] >= 20
        ]
        .sort_values(
            "cancellation_rate",
            ascending=False
        )
        .head(10)
        .copy()
    )



# HIGH CANCELLATION CUSTOMERS


def high_cancellation_customers(customer):

    return (
        customer[
            customer["total_bookings"] >= 3
        ]
        .sort_values(
            [
                "cancellation_rate",
                "cancelled_bookings"
            ],
            ascending=False
        )
        .head(20)
        .copy()
    )



# BUSINESS INSIGHTS


def generate_insights(
    summary,
    status,
    route,
    payment,
    vehicle,
    monthly,
    lead_time,
    customer
):

    print("\n")
    
    print("CANCELLATION & BOOKING BEHAVIOR INSIGHTS")
    

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n1. OVERALL BOOKING STATUS")

    print(
        f"Total Bookings       : "
        f"{summary['total_bookings']:,}"
    )

    print(
        f"Confirmed Bookings   : "
        f"{summary['confirmed_bookings']:,}"
    )

    print(
        f"Cancelled Bookings   : "
        f"{summary['cancelled_bookings']:,}"
    )

    print(
        f"Pending Bookings     : "
        f"{summary['pending_bookings']:,}"
    )

    print(
        f"Cancellation Rate    : "
        f"{summary['cancellation_rate']:.2f}%"
    )

    print(
        f"Cancelled Value      : "
        f"₹{summary['cancelled_revenue']:,.2f}"
    )

    print(
        f"Cancelled Value Share: "
        f"{summary['cancelled_value_share']:.2f}%"
    )

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    print("\n2. ROUTES WITH HIGHEST CANCELLATION")
    print("-" * 80)

    high_routes = high_cancellation_routes(
        route
    )

    if not high_routes.empty:

        print(
            high_routes[
                [
                    "route",
                    "total_bookings",
                    "cancelled_bookings",
                    "cancellation_rate",
                    "cancelled_value"
                ]
            ]
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Payment
    # --------------------------------------------------------

    print("\n3. CANCELLATION BY PAYMENT METHOD")
    print("-" * 80)

    print(
        payment.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    print("\n4. CANCELLATION BY VEHICLE TYPE")
    print("-" * 80)

    print(
        vehicle.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Lead time
    # --------------------------------------------------------

    print("\n5. CANCELLATION BY BOOKING LEAD TIME")
    print("-" * 80)

    print(
        lead_time.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    print("\n6. FREQUENT CANCELLERS")
    print("-" * 80)

    frequent = customer[
        customer["frequent_canceller"]
    ]

    print(
        f"Frequent cancellers: "
        f"{len(frequent):,}"
    )

    if not frequent.empty:

        print(
            frequent[
                [
                    "customer_id",
                    "total_bookings",
                    "cancelled_bookings",
                    "cancellation_rate"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    print("\n7. BUSINESS RECOMMENDATIONS")
    print("-" * 80)

    rate = summary[
        "cancellation_rate"
    ]

    if rate >= 25:

        print(
            "• Cancellation rate is high; investigate "
            "customer and operational cancellation drivers."
        )

    elif rate >= 15:

        print(
            "• Cancellation rate is moderate; monitor "
            "high-risk routes and booking patterns."
        )

    else:

        print(
            "• Cancellation rate is relatively low."
        )

    if not high_routes.empty:

        route_name = (
            high_routes.iloc[0]["route"]
        )

        print(
            f"• Investigate cancellation drivers on "
            f"{route_name}."
        )

    if not lead_time.empty:

        highest_lead = lead_time.loc[
            lead_time["cancellation_rate"].idxmax()
        ]

        print(
            f"• Highest cancellation behavior occurs "
            f"in the {highest_lead['lead_time_band']} "
            f"booking window."
        )

    print(
        "• Consider targeted reminders and flexible "
        "rescheduling options for high-risk bookings."
    )

    print(
        "• Monitor frequent cancellers separately "
        "from normal customers."
    )



# SAVE RESULTS


def save_results(
    status,
    route,
    payment,
    vehicle,
    monthly,
    weekday,
    lead_time,
    customer
):

    results = {

        "booking_status_distribution.csv":
            status,

        "route_cancellation.csv":
            route,

        "payment_cancellation.csv":
            payment,

        "vehicle_cancellation.csv":
            vehicle,

        "monthly_cancellation.csv":
            monthly,

        "weekday_cancellation.csv":
            weekday,

        "lead_time_cancellation.csv":
            lead_time,

        "customer_cancellation.csv":
            customer
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

    
    print("STEP 9.8 — CANCELLATION & BOOKING BEHAVIOR ANALYTICS")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    bookings, trips, routes = load_data()

    print(
        f"\nBookings loaded : "
        f"{len(bookings):,}"
    )

    print(
        f"Trips loaded    : "
        f"{len(trips):,}"
    )

    print(
        f"Routes loaded   : "
        f"{len(routes):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        bookings,
        trips,
        routes
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary = (
        overall_cancellation_summary(
            data
        )
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = status_distribution(
        data
    )

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    route = route_cancellation(
        data
    )

    # --------------------------------------------------------
    # Payment
    # --------------------------------------------------------

    payment = payment_cancellation(
        data
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    vehicle = vehicle_cancellation(
        data
    )

    # --------------------------------------------------------
    # Monthly
    # --------------------------------------------------------

    monthly = monthly_cancellation(
        data
    )

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    weekday = weekday_cancellation(
        data
    )

    # --------------------------------------------------------
    # Lead time
    # --------------------------------------------------------

    lead_time = lead_time_analysis(
        data
    )

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    customer = customer_cancellation(
        data
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_insights(
        summary,
        status,
        route,
        payment,
        vehicle,
        monthly,
        lead_time,
        customer
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        status,
        route,
        payment,
        vehicle,
        monthly,
        weekday,
        lead_time,
        customer
    )

    print("\n")
    
    print("STEP 9.8 COMPLETED")
    


if __name__ == "__main__":
    main()