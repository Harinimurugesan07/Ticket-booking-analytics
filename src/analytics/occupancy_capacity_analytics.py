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

    trips["arrival_datetime"] = pd.to_datetime(
        trips["arrival_datetime"],
        errors="coerce"
    )

    return bookings, trips, routes



# PREPARE TRIP CAPACITY DATA


def prepare_data(
    bookings,
    trips,
    routes
):

    # --------------------------------------------------------
    # Only confirmed bookings count toward seats sold
    # --------------------------------------------------------

    confirmed = bookings.copy()

    if "booking_status" in confirmed.columns:

        confirmed = confirmed[
            confirmed["booking_status"]
            .astype(str)
            .str.upper()
            .eq("CONFIRMED")
        ].copy()

    # --------------------------------------------------------
    # Aggregate seats sold per trip
    # --------------------------------------------------------

    trip_bookings = (
        confirmed
        .groupby("trip_id")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            seats_sold=(
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

    # --------------------------------------------------------
    # Merge trip information
    # --------------------------------------------------------

    data = trips.merge(
        trip_bookings,
        on="trip_id",
        how="left"
    )

    # Trips without bookings have zero seats sold
    data["bookings"] = (
        data["bookings"]
        .fillna(0)
    )

    data["seats_sold"] = (
        data["seats_sold"]
        .fillna(0)
    )

    data["revenue"] = (
        data["revenue"]
        .fillna(0)
    )

    data["unique_customers"] = (
        data["unique_customers"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Route information
    # --------------------------------------------------------

    route_columns = [
        "route_id",
        "route_name",
        "source",
        "destination",
        "distance_km"
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

    # --------------------------------------------------------
    # Available seats
    # --------------------------------------------------------

    data["available_seats"] = (
        data["total_seats"]
        -
        data["seats_sold"]
    )

    # --------------------------------------------------------
    # Occupancy percentage
    # --------------------------------------------------------

    data["occupancy_percentage"] = (
        data["seats_sold"]
        /
        data["total_seats"]
        *
        100
    )

    # --------------------------------------------------------
    # Prevent invalid occupancy
    # --------------------------------------------------------

    data["occupancy_percentage"] = (
        data["occupancy_percentage"]
        .clip(
            lower=0,
            upper=100
        )
    )

    # --------------------------------------------------------
    # Revenue per occupied seat
    # --------------------------------------------------------

    data["revenue_per_seat"] = 0.0

    mask = data["seats_sold"] > 0

    data.loc[
        mask,
        "revenue_per_seat"
    ] = (
        data.loc[mask, "revenue"]
        /
        data.loc[mask, "seats_sold"]
    )

    return data



# OCCUPANCY CLASSIFICATION


def classify_occupancy(data):

    def classify(value):

        if value < 30:

            return "Very Low"

        elif value < 50:

            return "Low"

        elif value < 75:

            return "Moderate"

        elif value < 90:

            return "High"

        else:

            return "Very High"

    data["occupancy_category"] = (
        data["occupancy_percentage"]
        .apply(classify)
    )

    return data



# OVERALL CAPACITY SUMMARY


def capacity_summary(data):

    total_trips = len(data)

    total_capacity = (
        data["total_seats"]
        .sum()
    )

    total_seats_sold = (
        data["seats_sold"]
        .sum()
    )

    total_available = (
        data["available_seats"]
        .sum()
    )

    total_revenue = (
        data["revenue"]
        .sum()
    )

    overall_occupancy = (
        total_seats_sold
        /
        total_capacity
        *
        100
        if total_capacity > 0
        else 0
    )

    average_trip_occupancy = (
        data["occupancy_percentage"]
        .mean()
    )

    return {
        "total_trips": total_trips,
        "total_capacity": total_capacity,
        "total_seats_sold": total_seats_sold,
        "total_available_seats": total_available,
        "total_revenue": total_revenue,
        "overall_occupancy": overall_occupancy,
        "average_trip_occupancy": average_trip_occupancy
    }



# ROUTE OCCUPANCY


def route_occupancy(data):

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
            trips=(
                "trip_id",
                "count"
            ),

            total_capacity=(
                "total_seats",
                "sum"
            ),

            seats_sold=(
                "seats_sold",
                "sum"
            ),

            available_seats=(
                "available_seats",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )
        .reset_index()
    )

    result["occupancy_percentage"] = (
        result["seats_sold"]
        /
        result["total_capacity"]
        *
        100
    )

    result["revenue_per_seat"] = (
        result["revenue"]
        /
        result["seats_sold"]
    )

    result = result.sort_values(
        "occupancy_percentage",
        ascending=False
    )

    return result



# VEHICLE TYPE OCCUPANCY


def vehicle_occupancy(data):

    result = (
        data
        .groupby("vehicle_type")
        .agg(
            trips=(
                "trip_id",
                "count"
            ),

            total_capacity=(
                "total_seats",
                "sum"
            ),

            seats_sold=(
                "seats_sold",
                "sum"
            ),

            available_seats=(
                "available_seats",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )
        .reset_index()
    )

    result["occupancy_percentage"] = (
        result["seats_sold"]
        /
        result["total_capacity"]
        *
        100
    )

    result["revenue_per_seat"] = (
        result["revenue"]
        /
        result["seats_sold"]
    )

    return result



# OCCUPANCY CATEGORY SUMMARY


def occupancy_categories(data):

    result = (
        data
        .groupby(
            "occupancy_category"
        )
        .agg(
            trips=(
                "trip_id",
                "count"
            ),

            total_capacity=(
                "total_seats",
                "sum"
            ),

            seats_sold=(
                "seats_sold",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )
        .reset_index()
    )

    result["occupancy_percentage"] = (
        result["seats_sold"]
        /
        result["total_capacity"]
        *
        100
    )

    return result



# UNDERUTILIZED TRIPS


def underutilized_trips(data):

    result = (
        data[
            data["occupancy_percentage"] < 30
        ]
        .copy()
        .sort_values(
            "occupancy_percentage"
        )
    )

    return result



# HIGH OCCUPANCY TRIPS


def high_occupancy_trips(data):

    result = (
        data[
            data["occupancy_percentage"] >= 90
        ]
        .copy()
        .sort_values(
            "occupancy_percentage",
            ascending=False
        )
    )

    return result



# PEAK CAPACITY DEMAND


def daily_capacity_analysis(data):

    data = data.copy()

    data["travel_day"] = (
        data["departure_datetime"]
        .dt.date
    )

    result = (
        data
        .groupby("travel_day")
        .agg(
            trips=(
                "trip_id",
                "count"
            ),

            total_capacity=(
                "total_seats",
                "sum"
            ),

            seats_sold=(
                "seats_sold",
                "sum"
            ),

            available_seats=(
                "available_seats",
                "sum"
            ),

            revenue=(
                "revenue",
                "sum"
            )
        )
        .reset_index()
    )

    result["occupancy_percentage"] = (
        result["seats_sold"]
        /
        result["total_capacity"]
        *
        100
    )

    result = result.sort_values(
        "seats_sold",
        ascending=False
    )

    return result



# BUSINESS INSIGHTS


def generate_insights(
    summary,
    route,
    vehicle,
    underutilized,
    high_occupancy,
    daily
):

    print("\n")
    
    print("CAPACITY & OCCUPANCY BUSINESS INSIGHTS")
    

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print("\n1. OVERALL CAPACITY")

    print(
        f"Total Trips             : "
        f"{summary['total_trips']:,}"
    )

    print(
        f"Total Seat Capacity     : "
        f"{summary['total_capacity']:,}"
    )

    print(
        f"Seats Sold              : "
        f"{summary['total_seats_sold']:,}"
    )

    print(
        f"Available Seats         : "
        f"{summary['total_available_seats']:,}"
    )

    print(
        f"Overall Occupancy      : "
        f"{summary['overall_occupancy']:.2f}%"
    )

    print(
        f"Average Trip Occupancy : "
        f"{summary['average_trip_occupancy']:.2f}%"
    )

    # --------------------------------------------------------
    # Top routes
    # --------------------------------------------------------

    print("\n2. HIGHEST OCCUPANCY ROUTES")
    print("-" * 80)

    print(
        route
        .head(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    print("\n3. OCCUPANCY BY VEHICLE TYPE")
    print("-" * 80)

    print(
        vehicle.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Underutilized
    # --------------------------------------------------------

    print("\n4. UNDERUTILIZED TRIPS")
    print("-" * 80)

    print(
        f"Trips below 30% occupancy: "
        f"{len(underutilized):,}"
    )

    if not underutilized.empty:

        print(
            underutilized[
                [
                    "trip_id",
                    "route",
                    "vehicle_type",
                    "total_seats",
                    "seats_sold",
                    "occupancy_percentage"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # High occupancy
    # --------------------------------------------------------

    print("\n5. HIGH-OCCUPANCY TRIPS")
    print("-" * 80)

    print(
        f"Trips at 90%+ occupancy: "
        f"{len(high_occupancy):,}"
    )

    if not high_occupancy.empty:

        print(
            high_occupancy[
                [
                    "trip_id",
                    "route",
                    "vehicle_type",
                    "total_seats",
                    "seats_sold",
                    "occupancy_percentage"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Peak capacity
    # --------------------------------------------------------

    print("\n6. PEAK CAPACITY DAYS")
    print("-" * 80)

    if not daily.empty:

        print(
            daily
            .head(10)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    print("\n7. BUSINESS RECOMMENDATIONS")
    print("-" * 80)

    occupancy = summary[
        "overall_occupancy"
    ]

    if occupancy < 40:

        print(
            "• Overall capacity utilization is low."
        )

        print(
            "• Consider reducing unnecessary trip capacity."
        )

        print(
            "• Use promotions to increase demand."
        )

    elif occupancy < 70:

        print(
            "• Capacity utilization is moderate."
        )

        print(
            "• Optimize schedules based on route-level demand."
        )

        print(
            "• Shift capacity toward stronger routes."
        )

    elif occupancy < 85:

        print(
            "• Capacity utilization is healthy."
        )

        print(
            "• Focus on optimizing high-demand periods."
        )

    else:

        print(
            "• Capacity utilization is very high."
        )

        print(
            "• Consider adding trips or larger vehicles."
        )

        print(
            "• Monitor high-demand routes for capacity shortages."
        )

    if len(underutilized) > 0:

        print(
            "• Investigate underutilized trips for "
            "schedule or capacity optimization."
        )

    if len(high_occupancy) > 0:

        print(
            "• Monitor near-full trips and consider "
            "additional capacity."
        )



# SAVE RESULTS


def save_results(
    trip_data,
    route,
    vehicle,
    categories,
    underutilized,
    high_occupancy,
    daily
):

    files = {

        "trip_occupancy.csv":
            trip_data,

        "route_occupancy.csv":
            route,

        "vehicle_occupancy.csv":
            vehicle,

        "occupancy_categories.csv":
            categories,

        "underutilized_trips.csv":
            underutilized,

        "high_occupancy_trips.csv":
            high_occupancy,

        "daily_capacity_analysis.csv":
            daily
    }

    print("\n")
    
    print("FILES SAVED")
    

    for filename, dataframe in files.items():

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

    
    print("STEP 9.5 — OCCUPANCY & CAPACITY ANALYTICS")
    

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
        f"Trips analyzed  : "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    data = classify_occupancy(
        data
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = capacity_summary(
        data
    )

    # --------------------------------------------------------
    # Route occupancy
    # --------------------------------------------------------

    route = route_occupancy(
        data
    )

    # --------------------------------------------------------
    # Vehicle occupancy
    # --------------------------------------------------------

    vehicle = vehicle_occupancy(
        data
    )

    # --------------------------------------------------------
    # Occupancy categories
    # --------------------------------------------------------

    categories = occupancy_categories(
        data
    )

    # --------------------------------------------------------
    # Underutilized
    # --------------------------------------------------------

    underutilized = underutilized_trips(
        data
    )

    # --------------------------------------------------------
    # High occupancy
    # --------------------------------------------------------

    high_occupancy = high_occupancy_trips(
        data
    )

    # --------------------------------------------------------
    # Daily
    # --------------------------------------------------------

    daily = daily_capacity_analysis(
        data
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print("\n")
    
    print("CAPACITY SUMMARY")
    

    print(
        f"Total Trips             : "
        f"{summary['total_trips']:,}"
    )

    print(
        f"Total Capacity          : "
        f"{summary['total_capacity']:,}"
    )

    print(
        f"Seats Sold              : "
        f"{summary['total_seats_sold']:,}"
    )

    print(
        f"Available Seats         : "
        f"{summary['total_available_seats']:,}"
    )

    print(
        f"Overall Occupancy       : "
        f"{summary['overall_occupancy']:.2f}%"
    )

    print(
        f"Average Trip Occupancy  : "
        f"{summary['average_trip_occupancy']:.2f}%"
    )

    # --------------------------------------------------------
    # Occupancy categories
    # --------------------------------------------------------

    print("\n")
    
    print("OCCUPANCY CATEGORIES")
    

    print(
        categories.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_insights(
        summary,
        route,
        vehicle,
        underutilized,
        high_occupancy,
        daily
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        data,
        route,
        vehicle,
        categories,
        underutilized,
        high_occupancy,
        daily
    )

    print("\n")
   
    print("STEP 9.5 COMPLETED")
   


if __name__ == "__main__":
    main()