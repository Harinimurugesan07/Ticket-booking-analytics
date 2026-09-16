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

    return bookings, trips, routes



# PREPARE DATA


def prepare_data(
    bookings,
    trips,
    routes
):

    data = bookings.copy()

    # --------------------------------------------------------
    # Confirmed bookings only
    # --------------------------------------------------------

    data = data[
        data["booking_status"]
        .astype(str)
        .str.upper()
        .eq("CONFIRMED")
    ].copy()

    # --------------------------------------------------------
    # Join bookings → trips
    # --------------------------------------------------------

    trip_columns = [
        "trip_id",
        "route_id",
        "vehicle_type",
        "total_seats",
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
    # Join trips → routes
    # --------------------------------------------------------

    route_columns = [
        "route_id",
        "route_name",
        "source",
        "destination",
        "distance_km",
        "base_fare"
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
    # Revenue per ticket
    # --------------------------------------------------------

    data["revenue_per_ticket"] = (
        data["total_amount"]
        /
        data["seat_count"]
    )

    # --------------------------------------------------------
    # Price difference from base fare
    # --------------------------------------------------------

    data["price_difference"] = (
        data["revenue_per_ticket"]
        -
        data["base_fare"]
    )

    # --------------------------------------------------------
    # Price premium percentage
    # --------------------------------------------------------

    data["price_premium_percentage"] = (
        (
            data["revenue_per_ticket"]
            -
            data["base_fare"]
        )
        /
        data["base_fare"]
        *
        100
    )

    return data



# OVERALL REVENUE SUMMARY


def overall_summary(data):

    total_revenue = (
        data["total_amount"]
        .sum()
    )

    total_bookings = len(data)

    total_tickets = (
        data["seat_count"]
        .sum()
    )

    average_booking_value = (
        data["total_amount"]
        .mean()
    )

    average_ticket_revenue = (
        data["revenue_per_ticket"]
        .mean()
    )

    return {
        "total_revenue": total_revenue,
        "total_bookings": total_bookings,
        "total_tickets": total_tickets,
        "average_booking_value":
            average_booking_value,
        "average_ticket_revenue":
            average_ticket_revenue
    }



# ROUTE REVENUE ANALYSIS


def route_revenue(data):

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
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            average_ticket_price=(
                "revenue_per_ticket",
                "mean"
            ),

            average_booking_value=(
                "total_amount",
                "mean"
            ),

            base_fare=(
                "base_fare",
                "mean"
            ),

            distance_km=(
                "distance_km",
                "mean"
            )
        )
        .reset_index()
    )

    result["revenue_per_ticket"] = (
        result["revenue"]
        /
        result["tickets_sold"]
    )

    result["price_premium_percentage"] = (
        (
            result["revenue_per_ticket"]
            -
            result["base_fare"]
        )
        /
        result["base_fare"]
        *
        100
    )

    result["revenue_rank"] = (
        result["revenue"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    return result.sort_values(
        "revenue",
        ascending=False
    )



# VEHICLE TYPE REVENUE


def vehicle_revenue(data):

    result = (
        data
        .groupby("vehicle_type")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            average_ticket_price=(
                "revenue_per_ticket",
                "mean"
            )
        )
        .reset_index()
    )

    result["revenue_per_ticket"] = (
        result["revenue"]
        /
        result["tickets_sold"]
    )

    result["revenue_share"] = (
        result["revenue"]
        /
        result["revenue"].sum()
        *
        100
    )

    return result.sort_values(
        "revenue",
        ascending=False
    )



# PAYMENT METHOD REVENUE


def payment_revenue(data):

    result = (
        data
        .groupby("payment_method")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
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

    result["average_booking_value"] = (
        result["revenue"]
        /
        result["bookings"]
    )

    result["revenue_share"] = (
        result["revenue"]
        /
        result["revenue"].sum()
        *
        100
    )

    return result.sort_values(
        "revenue",
        ascending=False
    )



# PRICE BAND ANALYSIS


def price_band_analysis(data):

    data = data.copy()

    # --------------------------------------------------------
    # Create price bands based on ticket revenue
    # --------------------------------------------------------

    data["price_band"] = pd.cut(
        data["revenue_per_ticket"],
        bins=[
            0,
            300,
            500,
            700,
            1000,
            float("inf")
        ],
        labels=[
            "Below ₹300",
            "₹300–₹500",
            "₹500–₹700",
            "₹700–₹1000",
            "Above ₹1000"
        ],
        include_lowest=True
    )

    result = (
        data
        .groupby(
            "price_band",
            observed=True
        )
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            average_price=(
                "revenue_per_ticket",
                "mean"
            )
        )
        .reset_index()
    )

    result["revenue_share"] = (
        result["revenue"]
        /
        result["revenue"].sum()
        *
        100
    )

    return result



# MONTHLY REVENUE


def monthly_revenue(data):

    data = data.copy()

    data["booking_month"] = (
        data["booking_date"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        data
        .groupby("booking_month")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            average_booking_value=(
                "total_amount",
                "mean"
            )
        )
        .reset_index()
    )

    return result.sort_values(
        "booking_month"
    )



# DAILY REVENUE


def daily_revenue(data):

    data = data.copy()

    data["booking_day"] = (
        data["booking_date"]
        .dt.date
    )

    result = (
        data
        .groupby("booking_day")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
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

    return result.sort_values(
        "booking_day"
    )



# HIGH VALUE ROUTES


def high_value_routes(route):

    return (
        route
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(10)
        .copy()
    )



# LOW REVENUE ROUTES


def low_revenue_routes(route):

    return (
        route
        .sort_values(
            "revenue",
            ascending=True
        )
        .head(10)
        .copy()
    )



# PRICE PREMIUM ANALYSIS


def price_premium_analysis(data):

    result = (
        data
        .groupby("route")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets_sold=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            average_ticket_price=(
                "revenue_per_ticket",
                "mean"
            ),

            average_base_fare=(
                "base_fare",
                "mean"
            ),

            average_price_premium=(
                "price_premium_percentage",
                "mean"
            )
        )
        .reset_index()
    )

    return result.sort_values(
        "average_price_premium",
        ascending=False
    )



# BUSINESS INSIGHTS


def generate_insights(
    summary,
    route,
    vehicle,
    payment,
    price_bands,
    premium
):

    print("\n")
    
    print("REVENUE & PRICING BUSINESS INSIGHTS")
    

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n1. OVERALL REVENUE")

    print(
        f"Total Revenue          : "
        f"₹{summary['total_revenue']:,.2f}"
    )

    print(
        f"Total Confirmed Bookings: "
        f"{summary['total_bookings']:,}"
    )

    print(
        f"Total Tickets Sold     : "
        f"{summary['total_tickets']:,}"
    )

    print(
        f"Average Booking Value  : "
        f"₹{summary['average_booking_value']:,.2f}"
    )

    print(
        f"Average Ticket Revenue : "
        f"₹{summary['average_ticket_revenue']:,.2f}"
    )

    # --------------------------------------------------------
    # Top routes
    # --------------------------------------------------------

    print("\n2. TOP REVENUE ROUTES")
    print("-" * 80)

    print(
        route[
            [
                "route",
                "bookings",
                "tickets_sold",
                "revenue",
                "revenue_per_ticket",
                "price_premium_percentage"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    print("\n3. REVENUE BY VEHICLE TYPE")
    print("-" * 80)

    print(
        vehicle.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Payment
    # --------------------------------------------------------

    print("\n4. REVENUE BY PAYMENT METHOD")
    print("-" * 80)

    print(
        payment.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Price bands
    # --------------------------------------------------------

    print("\n5. PRICE BAND PERFORMANCE")
    print("-" * 80)

    print(
        price_bands.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

    print("\n6. ROUTE PRICE PREMIUM")
    print("-" * 80)

    print(
        premium
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Business recommendations
    # --------------------------------------------------------

    print("\n7. BUSINESS RECOMMENDATIONS")
    print("-" * 80)

    if not route.empty:

        top_route = route.iloc[0]

        print(
            f"• Prioritize the highest-revenue route: "
            f"{top_route['route']}."
        )

    if not vehicle.empty:

        top_vehicle = vehicle.iloc[0]

        print(
            f"• {top_vehicle['vehicle_type']} generates "
            f"the highest total revenue."
        )

    print(
        "• Monitor ticket prices against base fares."
    )

    print(
        "• Evaluate whether high-price routes maintain "
        "sufficient demand."
    )

    print(
        "• Use route-level revenue when deciding where "
        "to increase capacity."
    )

    print(
        "• Review low-revenue routes for pricing and "
        "demand optimization opportunities."
    )



# SAVE RESULTS


def save_results(
    route,
    vehicle,
    payment,
    price_bands,
    monthly,
    daily,
    premium
):

    results = {

        "route_revenue.csv":
            route,

        "vehicle_revenue.csv":
            vehicle,

        "payment_method_revenue.csv":
            payment,

        "price_band_analysis.csv":
            price_bands,

        "monthly_revenue.csv":
            monthly,

        "daily_revenue.csv":
            daily,

        "route_price_premium.csv":
            premium
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

    
    print("STEP 9.6 — REVENUE & PRICING ANALYTICS")
    

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

    print(
        f"Confirmed bookings: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    summary = overall_summary(
        data
    )

    route = route_revenue(
        data
    )

    vehicle = vehicle_revenue(
        data
    )

    payment = payment_revenue(
        data
    )

    price_bands = price_band_analysis(
        data
    )

    monthly = monthly_revenue(
        data
    )

    daily = daily_revenue(
        data
    )

    premium = price_premium_analysis(
        data
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_insights(
        summary,
        route,
        vehicle,
        payment,
        price_bands,
        premium
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        route,
        vehicle,
        payment,
        price_bands,
        monthly,
        daily,
        premium
    )

    print("\n")
    
    print("STEP 9.6 COMPLETED")
    


if __name__ == "__main__":
    main()