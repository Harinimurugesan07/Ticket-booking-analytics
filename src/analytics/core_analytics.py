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

    customers["registration_date"] = pd.to_datetime(
        customers["registration_date"]
    )

    trips["departure_datetime"] = pd.to_datetime(
        trips["departure_datetime"]
    )

    trips["arrival_datetime"] = pd.to_datetime(
        trips["arrival_datetime"]
    )

    bookings["booking_date"] = pd.to_datetime(
        bookings["booking_date"]
    )

    bookings["travel_date"] = pd.to_datetime(
        bookings["travel_date"]
    )

    return (
        customers,
        operators,
        routes,
        trips,
        bookings
    )



# 1. OVERALL BOOKING KPIs


def booking_kpis(bookings):

    total_bookings = len(bookings)

    total_tickets = bookings[
        "seat_count"
    ].sum()

    confirmed_bookings = (
        bookings["booking_status"] == "CONFIRMED"
    ).sum()

    cancelled_bookings = (
        bookings["booking_status"] == "CANCELLED"
    ).sum()

    pending_bookings = (
        bookings["booking_status"] == "PENDING"
    ).sum()

    cancellation_rate = (
        cancelled_bookings /
        total_bookings *
        100
    )

    confirmation_rate = (
        confirmed_bookings /
        total_bookings *
        100
    )

    return {
        "total_bookings": total_bookings,
        "total_tickets": total_tickets,
        "confirmed_bookings": confirmed_bookings,
        "cancelled_bookings": cancelled_bookings,
        "pending_bookings": pending_bookings,
        "cancellation_rate": cancellation_rate,
        "confirmation_rate": confirmation_rate
    }



# 2. REVENUE ANALYTICS


def revenue_kpis(bookings):

    total_revenue = bookings[
        "total_amount"
    ].sum()

    confirmed_revenue = bookings.loc[
        bookings["booking_status"] == "CONFIRMED",
        "total_amount"
    ].sum()

    cancelled_revenue = bookings.loc[
        bookings["booking_status"] == "CANCELLED",
        "total_amount"
    ].sum()

    average_booking_value = bookings[
        "total_amount"
    ].mean()

    average_ticket_price = bookings[
        "ticket_price"
    ].mean()

    return {
        "total_revenue": total_revenue,
        "confirmed_revenue": confirmed_revenue,
        "cancelled_revenue": cancelled_revenue,
        "average_booking_value": average_booking_value,
        "average_ticket_price": average_ticket_price
    }



# 3. PAYMENT ANALYTICS


def payment_analytics(bookings):

    result = (
        bookings
        .groupby("payment_method")
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            revenue=("total_amount", "sum")
        )
        .sort_values(
            "revenue",
            ascending=False
        )
        .reset_index()
    )

    return result



# 4. VEHICLE ANALYTICS


def vehicle_analytics(
    bookings,
    trips
):

    data = bookings.merge(
        trips[
            [
                "trip_id",
                "vehicle_type",
                "total_seats"
            ]
        ],
        on="trip_id",
        how="left"
    )

    result = (
        data
        .groupby("vehicle_type")
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            revenue=("total_amount", "sum"),
            average_ticket_price=(
                "ticket_price",
                "mean"
            )
        )
        .sort_values(
            "revenue",
            ascending=False
        )
        .reset_index()
    )

    return result



# 5. ROUTE ANALYTICS


def route_analytics(
    bookings,
    trips,
    routes
):

    data = bookings.merge(
        trips[
            [
                "trip_id",
                "route_id"
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
                "distance_km",
                "base_fare"
            ]
        ],
        on="route_id",
        how="left"
    )

    result = (
        data
        .groupby(
            [
                "route_id",
                "route_name",
                "source",
                "destination"
            ]
        )
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            revenue=("total_amount", "sum"),
            average_ticket_price=(
                "ticket_price",
                "mean"
            )
        )
        .sort_values(
            "revenue",
            ascending=False
        )
        .reset_index()
    )

    return result



# 6. OPERATOR ANALYTICS


def operator_analytics(
    bookings,
    trips,
    operators
):

    data = bookings.merge(
        trips[
            [
                "trip_id",
                "operator_id"
            ]
        ],
        on="trip_id",
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

    result = (
        data
        .groupby(
            [
                "operator_id",
                "operator_name"
            ]
        )
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            revenue=("total_amount", "sum")
        )
        .sort_values(
            "revenue",
            ascending=False
        )
        .reset_index()
    )

    return result



# 7. CUSTOMER ANALYTICS


def customer_analytics(
    bookings,
    customers
):

    result = (
        bookings
        .groupby("customer_id")
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            total_spend=("total_amount", "sum"),
            average_booking_value=(
                "total_amount",
                "mean"
            )
        )
        .reset_index()
    )

    result = result.merge(
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

    result = result.sort_values(
        "total_spend",
        ascending=False
    )

    return result



# 8. CANCELLATION ANALYTICS


def cancellation_analytics(bookings):

    total = len(bookings)

    cancelled = bookings[
        bookings["booking_status"] ==
        "CANCELLED"
    ]

    cancellation_rate = (
        len(cancelled) /
        total *
        100
    )

    cancellation_by_payment = (
        pd.crosstab(
            bookings["payment_method"],
            bookings["booking_status"]
        )
        .reset_index()
    )

    cancellation_by_payment["cancellation_rate"] = (
        cancellation_by_payment.get(
            "CANCELLED",
            0
        ) /
        cancellation_by_payment[
            [
                column
                for column in [
                    "CONFIRMED",
                    "CANCELLED",
                    "PENDING"
                ]
                if column in cancellation_by_payment.columns
            ]
        ].sum(axis=1)
        * 100
    )

    return {
        "overall_cancellation_rate":
            cancellation_rate,
        "cancelled_bookings":
            len(cancelled),
        "cancelled_revenue":
            cancelled["total_amount"].sum(),
        "by_payment_method":
            cancellation_by_payment
    }



# 9. DAILY BOOKING ANALYTICS


def daily_booking_analytics(bookings):

    result = (
        bookings
        .groupby("booking_date")
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            revenue=("total_amount", "sum")
        )
        .reset_index()
        .sort_values("booking_date")
    )

    return result



# 10. MONTHLY BOOKING ANALYTICS


def monthly_booking_analytics(bookings):

    data = bookings.copy()

    data["month"] = (
        data["booking_date"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        data
        .groupby("month")
        .agg(
            bookings=("booking_id", "count"),
            tickets=("seat_count", "sum"),
            revenue=("total_amount", "sum")
        )
        .reset_index()
    )

    return result



# 11. OCCUPANCY ANALYTICS


def occupancy_analytics(
    bookings,
    trips
):

    # Only confirmed bookings should count
    # toward actual occupied seats.

    confirmed = bookings[
        bookings["booking_status"] ==
        "CONFIRMED"
    ]

    booked_seats = (
        confirmed
        .groupby("trip_id")["seat_count"]
        .sum()
        .reset_index(
            name="booked_seats"
        )
    )

    result = trips[
        [
            "trip_id",
            "route_id",
            "vehicle_type",
            "total_seats",
            "departure_datetime"
        ]
    ].merge(
        booked_seats,
        on="trip_id",
        how="left"
    )

    result["booked_seats"] = (
        result["booked_seats"]
        .fillna(0)
    )

    result["occupancy_rate"] = (
        result["booked_seats"] /
        result["total_seats"] *
        100
    )

    result["overbooked"] = (
        result["booked_seats"] >
        result["total_seats"]
    )

    return result



# 12. PEAK BOOKING ANALYTICS


def peak_booking_analytics(bookings):

    data = bookings.copy()

    data["booking_day"] = (
        data["booking_date"]
        .dt.day_name()
    )

    data["booking_hour"] = (
        data["booking_date"]
        .dt.hour
    )

    day_analysis = (
        data
        .groupby("booking_day")
        .size()
        .sort_values(
            ascending=False
        )
    )

    return {
        "by_day": day_analysis
    }



# MAIN ANALYTICS REPORT


def main():

    
    print("TICKET BOOKING ANALYTICS ENGINE")
    

    (
        customers,
        operators,
        routes,
        trips,
        bookings
    ) = load_data()

    # --------------------------------------------------------
    # BOOKING KPIs
    # --------------------------------------------------------

    booking_results = booking_kpis(
        bookings
    )

    print("\n")
    
    print("BOOKING KPIs")
    

    for key, value in booking_results.items():

        if "rate" in key:
            print(
                f"{key:<30}: {value:.2f}%"
            )
        else:
            print(
                f"{key:<30}: {value:,.2f}"
            )

    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    revenue_results = revenue_kpis(
        bookings
    )

    print("\n")
    
    print("REVENUE KPIs")
    

    for key, value in revenue_results.items():

        print(
            f"{key:<30}: ₹{value:,.2f}"
        )

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    print("\n")
    
    print("PAYMENT ANALYTICS")
    

    print(
        payment_analytics(
            bookings
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # VEHICLES
    # --------------------------------------------------------

    print("\n")
    
    print("VEHICLE ANALYTICS")
    

    print(
        vehicle_analytics(
            bookings,
            trips
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # ROUTES
    # --------------------------------------------------------

    print("\n")
    
    print("TOP 10 ROUTES BY REVENUE")
    

    routes_result = route_analytics(
        bookings,
        trips,
        routes
    )

    print(
        routes_result
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # OPERATORS
    # --------------------------------------------------------

    print("\n")
    
    print("OPERATOR ANALYTICS")
    

    print(
        operator_analytics(
            bookings,
            trips,
            operators
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # CUSTOMERS
    # --------------------------------------------------------

    print("\n")
    
    print("TOP 10 CUSTOMERS BY SPENDING")
    

    print(
        customer_analytics(
            bookings,
            customers
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # CANCELLATIONS
    # --------------------------------------------------------

    cancellation_results = (
        cancellation_analytics(
            bookings
        )
    )

    print("\n")
    
    print("CANCELLATION ANALYTICS")
    

    print(
        f"Cancellation Rate : "
        f"{cancellation_results['overall_cancellation_rate']:.2f}%"
    )

    print(
        f"Cancelled Bookings: "
        f"{cancellation_results['cancelled_bookings']:,}"
    )

    print(
        f"Cancelled Revenue : "
        f"₹{cancellation_results['cancelled_revenue']:,.2f}"
    )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    print("\n")
    
    print("MONTHLY BOOKING ANALYTICS")
    

    print(
        monthly_booking_analytics(
            bookings
        ).to_string(index=False)
    )

    # --------------------------------------------------------
    # OCCUPANCY
    # --------------------------------------------------------

    occupancy = occupancy_analytics(
        bookings,
        trips
    )

    print("\n")
    
    print("OCCUPANCY ANALYTICS")
    

    average_occupancy = (
        occupancy["occupancy_rate"]
        .mean()
    )

    overbooked_trips = (
        occupancy["overbooked"]
        .sum()
    )

    print(
        f"Average Occupancy : "
        f"{average_occupancy:.2f}%"
    )

    print(
        f"Overbooked Trips  : "
        f"{overbooked_trips:,}"
    )

    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print("\n")
    
    print("CORE ANALYTICS COMPLETED")
    


if __name__ == "__main__":
    main()