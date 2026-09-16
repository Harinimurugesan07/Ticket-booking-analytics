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

    bookings_file = os.path.join(
        DATA_DIR,
        "bookings_clean.csv"
    )

    trips_file = os.path.join(
        DATA_DIR,
        "trips_clean.csv"
    )

    routes_file = os.path.join(
        DATA_DIR,
        "routes_clean.csv"
    )

    bookings = pd.read_csv(
        bookings_file
    )

    trips = pd.read_csv(
        trips_file
    )

    routes = pd.read_csv(
        routes_file
    )

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

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

    trips["arrival_datetime"] = pd.to_datetime(
        trips["arrival_datetime"],
        errors="coerce"
    )

    return bookings, trips, routes



# PREPARE ROUTE-LEVEL BOOKING DATA


def prepare_data(
    bookings,
    trips,
    routes
):

    data = bookings.copy()

    # --------------------------------------------------------
    # Keep confirmed bookings only
    # --------------------------------------------------------

    if "booking_status" in data.columns:

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
        "operator_id",
        "vehicle_type",
        "departure_datetime",
        "arrival_datetime",
        "total_seats"
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
    # Create route label
    # --------------------------------------------------------

    data["route"] = (
        data["source"]
        .astype(str)
        .str.strip()
        +
        " → "
        +
        data["destination"]
        .astype(str)
        .str.strip()
    )

    return data



# CHECK JOIN QUALITY


def validate_joins(data):

    print("\n")
    
    print("JOIN VALIDATION")
    

    total_rows = len(data)

    missing_trip_routes = (
        data["route_id"]
        .isna()
        .sum()
    )

    missing_route_names = (
        data["route"]
        .isin(
            [
                "nan → nan",
                "None → None"
            ]
        )
        .sum()
    )

    print(
        f"Rows after joins          : {total_rows:,}"
    )

    print(
        f"Missing route_id          : "
        f"{missing_trip_routes:,}"
    )

    print(
        f"Missing route information : "
        f"{missing_route_names:,}"
    )

    if total_rows > 0:

        valid_rows = (
            total_rows
            -
            missing_trip_routes
        )

        join_rate = (
            valid_rows
            /
            total_rows
            *
            100
        )

        print(
            f"Route join success rate   : "
            f"{join_rate:.2f}%"
        )



# ROUTE PERFORMANCE


def route_performance(data):

    route = (
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

            unique_customers=(
                "customer_id",
                "nunique"
            ),

            average_booking_value=(
                "total_amount",
                "mean"
            ),

            average_tickets_per_booking=(
                "seat_count",
                "mean"
            ),

            average_distance_km=(
                "distance_km",
                "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Revenue per ticket
    # --------------------------------------------------------

    route["revenue_per_ticket"] = (
        route["revenue"]
        /
        route["tickets_sold"]
    )

    # --------------------------------------------------------
    # Revenue per customer
    # --------------------------------------------------------

    route["revenue_per_customer"] = (
        route["revenue"]
        /
        route["unique_customers"]
    )

    # --------------------------------------------------------
    # Demand rank
    # --------------------------------------------------------

    route["demand_rank"] = (
        route["tickets_sold"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Revenue rank
    # --------------------------------------------------------

    route["revenue_rank"] = (
        route["revenue"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    return route



# ROUTE DEMAND CLASSIFICATION


def classify_route_demand(route):

    if len(route) == 0:

        route["demand_category"] = []

        return route

    low_threshold = (
        route["tickets_sold"]
        .quantile(0.33)
    )

    high_threshold = (
        route["tickets_sold"]
        .quantile(0.67)
    )

    def classify(value):

        if value >= high_threshold:

            return "High Demand"

        elif value <= low_threshold:

            return "Low Demand"

        else:

            return "Moderate Demand"

    route["demand_category"] = (
        route["tickets_sold"]
        .apply(classify)
    )

    return route



# REVENUE SHARE


def calculate_revenue_share(route):

    total_revenue = (
        route["revenue"]
        .sum()
    )

    if total_revenue > 0:

        route["revenue_share"] = (
            route["revenue"]
            /
            total_revenue
            *
            100
        )

    else:

        route["revenue_share"] = 0

    return route



# PEAK TRAVEL DATES


def peak_travel_dates(data):

    if "travel_date" not in data.columns:

        return pd.DataFrame()

    daily = (
        data
        .groupby(
            data["travel_date"].dt.date
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

            unique_customers=(
                "customer_id",
                "nunique"
            )
        )
        .reset_index()
    )

    daily = daily.rename(
        columns={
            "travel_date":
                "travel_day"
        }
    )

    daily = daily.sort_values(
        "tickets_sold",
        ascending=False
    )

    return daily



# ROUTE + VEHICLE TYPE ANALYSIS


def route_vehicle_analysis(data):

    result = (
        data
        .groupby(
            [
                "route",
                "vehicle_type"
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
            )
        )
        .reset_index()
    )

    result["average_booking_value"] = (
        result["revenue"]
        /
        result["bookings"]
    )

    result = result.sort_values(
        "tickets_sold",
        ascending=False
    )

    return result



# SOURCE → DESTINATION ANALYSIS


def source_destination_analysis(data):

    result = (
        data
        .groupby(
            [
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

            customers=(
                "customer_id",
                "nunique"
            )
        )
        .reset_index()
    )

    result["average_booking_value"] = (
        result["revenue"]
        /
        result["bookings"]
    )

    result = result.sort_values(
        "tickets_sold",
        ascending=False
    )

    return result



# ROUTE BUSINESS INSIGHTS


def generate_insights(
    route,
    daily,
    route_vehicle
):

    print("\n")
    
    print("ROUTE & DEMAND BUSINESS INSIGHTS")
    

    # --------------------------------------------------------
    # HIGH DEMAND ROUTES
    # --------------------------------------------------------

    high_demand = (
        route[
            route["demand_category"]
            ==
            "High Demand"
        ]
        .sort_values(
            "tickets_sold",
            ascending=False
        )
        .head(5)
    )

    print("\n1. TOP HIGH-DEMAND ROUTES")
    print("-" * 80)

    for _, row in high_demand.iterrows():

        print(
            f"{row['route']} | "
            f"Tickets: {int(row['tickets_sold']):,} | "
            f"Bookings: {int(row['bookings']):,} | "
            f"Revenue: ₹{row['revenue']:,.2f}"
        )

    # --------------------------------------------------------
    # LOW DEMAND ROUTES
    # --------------------------------------------------------

    low_demand = (
        route[
            route["demand_category"]
            ==
            "Low Demand"
        ]
        .sort_values(
            "tickets_sold",
            ascending=True
        )
        .head(5)
    )

    print("\n2. LOW-DEMAND ROUTES")
    print("-" * 80)

    for _, row in low_demand.iterrows():

        print(
            f"{row['route']} | "
            f"Tickets: {int(row['tickets_sold']):,} | "
            f"Bookings: {int(row['bookings']):,} | "
            f"Revenue: ₹{row['revenue']:,.2f}"
        )

    # --------------------------------------------------------
    # REVENUE LEADERS
    # --------------------------------------------------------

    revenue_leaders = (
        route
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(5)
    )

    print("\n3. TOP ROUTES BY REVENUE")
    print("-" * 80)

    for _, row in revenue_leaders.iterrows():

        print(
            f"{row['route']} | "
            f"Revenue: ₹{row['revenue']:,.2f} | "
            f"Revenue Share: {row['revenue_share']:.2f}%"
        )

    # --------------------------------------------------------
    # PEAK TRAVEL DATES
    # --------------------------------------------------------

    print("\n4. TOP 10 PEAK TRAVEL DATES")
    print("-" * 80)

    if daily.empty:

        print(
            "Travel date information unavailable."
        )

    else:

        print(
            daily
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # VEHICLE TYPE
    # --------------------------------------------------------

    print("\n5. ROUTE + VEHICLE TYPE DEMAND")
    print("-" * 80)

    print(
        route_vehicle
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    print("\n6. BUSINESS RECOMMENDATIONS")
    print("-" * 80)

    print(
        "• Increase capacity on high-demand routes."
    )

    print(
        "• Consider additional trips during peak dates."
    )

    print(
        "• Use promotions to improve low-demand routes."
    )

    print(
        "• Review pricing on routes with low demand."
    )

    print(
        "• Match vehicle capacity with route demand."
    )

    print(
        "• Prioritize high-revenue routes for expansion."
    )



# SAVE RESULTS


def save_results(
    route,
    daily,
    route_vehicle,
    source_destination
):

    route_file = os.path.join(
        OUTPUT_DIR,
        "route_performance.csv"
    )

    daily_file = os.path.join(
        OUTPUT_DIR,
        "peak_travel_dates.csv"
    )

    vehicle_file = os.path.join(
        OUTPUT_DIR,
        "route_vehicle_demand.csv"
    )

    source_destination_file = os.path.join(
        OUTPUT_DIR,
        "source_destination_demand.csv"
    )

    route.to_csv(
        route_file,
        index=False
    )

    daily.to_csv(
        daily_file,
        index=False
    )

    route_vehicle.to_csv(
        vehicle_file,
        index=False
    )

    source_destination.to_csv(
        source_destination_file,
        index=False
    )

    print("\n")
    
    print("FILES SAVED")
    

    print(
        f"Route Performance:\n{route_file}"
    )

    print(
        f"\nPeak Travel Dates:\n{daily_file}"
    )

    print(
        f"\nRoute Vehicle Demand:\n{vehicle_file}"
    )

    print(
        f"\nSource Destination Demand:\n"
        f"{source_destination_file}"
    )



# MAIN


def main():

    
    print("STEP 9.4 — ROUTE & DEMAND ANALYTICS")
    

    # --------------------------------------------------------
    # LOAD
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
    # PREPARE
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

    print(
        f"Routes identified: "
        f"{data['route_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # VALIDATE JOINS
    # --------------------------------------------------------

    validate_joins(
        data
    )

    # --------------------------------------------------------
    # ROUTE PERFORMANCE
    # --------------------------------------------------------

    route = route_performance(
        data
    )

    # --------------------------------------------------------
    # DEMAND CLASSIFICATION
    # --------------------------------------------------------

    route = classify_route_demand(
        route
    )

    # --------------------------------------------------------
    # REVENUE SHARE
    # --------------------------------------------------------

    route = calculate_revenue_share(
        route
    )

    # --------------------------------------------------------
    # PEAK TRAVEL DATES
    # --------------------------------------------------------

    daily = peak_travel_dates(
        data
    )

    # --------------------------------------------------------
    # ROUTE + VEHICLE
    # --------------------------------------------------------

    route_vehicle = route_vehicle_analysis(
        data
    )

    # --------------------------------------------------------
    # SOURCE DESTINATION
    # --------------------------------------------------------

    source_destination = (
        source_destination_analysis(
            data
        )
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    
    print("ROUTE PERFORMANCE")
    

    display_columns = [
        "route_id",
        "route",
        "bookings",
        "tickets_sold",
        "revenue",
        "unique_customers",
        "average_booking_value",
        "revenue_per_ticket",
        "revenue_share",
        "demand_category"
    ]

    print(
        route[
            display_columns
        ]
        .sort_values(
            "tickets_sold",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    generate_insights(
        route,
        daily,
        route_vehicle
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        route,
        daily,
        route_vehicle,
        source_destination
    )

    print("\n")
    
    print("STEP 9.4 COMPLETED")
    


if __name__ == "__main__":
    main()