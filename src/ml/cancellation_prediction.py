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

ML_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ml"
)

os.makedirs(
    ML_DIR,
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

    return bookings, trips, routes



# PREPARE ML DATASET


def prepare_ml_dataset(
    bookings,
    trips,
    routes
):

    data = bookings.copy()

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    data["booking_date"] = pd.to_datetime(
        data["booking_date"],
        errors="coerce"
    )

    data["travel_date"] = pd.to_datetime(
        data["travel_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Merge trip information
    # --------------------------------------------------------

    trip_columns = [
        "trip_id",
        "route_id",
        "vehicle_type",
        "departure_datetime",
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
    # Merge route information
    # --------------------------------------------------------

    route_columns = [
        "route_id",
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
    # TARGET
    # --------------------------------------------------------

    data["is_cancelled"] = (
        data["booking_status"]
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("CANCELLED")
        .astype(int)
    )

    # --------------------------------------------------------
    # FEATURE: LEAD TIME
    # --------------------------------------------------------

    data["lead_time_days"] = (
        data["travel_date"]
        -
        data["booking_date"]
    ).dt.days

    # --------------------------------------------------------
    # FEATURE: BOOKING MONTH
    # --------------------------------------------------------

    data["booking_month"] = (
        data["booking_date"]
        .dt.month
    )

    # --------------------------------------------------------
    # FEATURE: TRAVEL MONTH
    # --------------------------------------------------------

    data["travel_month"] = (
        data["travel_date"]
        .dt.month
    )

    # --------------------------------------------------------
    # FEATURE: BOOKING DAY OF WEEK
    # --------------------------------------------------------

    data["booking_day_of_week"] = (
        data["booking_date"]
        .dt.dayofweek
    )

    # --------------------------------------------------------
    # FEATURE: TRAVEL DAY OF WEEK
    # --------------------------------------------------------

    data["travel_day_of_week"] = (
        data["travel_date"]
        .dt.dayofweek
    )

    # --------------------------------------------------------
    # FEATURE: BOOKING HOUR
    # --------------------------------------------------------

    data["booking_hour"] = (
        data["booking_date"]
        .dt.hour
    )

    # --------------------------------------------------------
    # FEATURE: TRAVEL HOUR
    # --------------------------------------------------------

    data["travel_hour"] = (
        pd.to_datetime(
            data["departure_datetime"],
            errors="coerce"
        )
        .dt.hour
    )

    # --------------------------------------------------------
    # FEATURE: PRICE PER SEAT
    # --------------------------------------------------------

    data["price_per_seat"] = (
        data["total_amount"]
        /
        data["seat_count"].replace(
            0,
            1
        )
    )

    # --------------------------------------------------------
    # FEATURE: DISTANCE CATEGORY
    # --------------------------------------------------------

    data["distance_category"] = pd.cut(
        data["distance_km"],
        bins=[
            -1,
            100,
            300,
            600,
            float("inf")
        ],
        labels=[
            "Short",
            "Medium",
            "Long",
            "Very Long"
        ]
    )

    # --------------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------------

    feature_columns = [
        "customer_id",
        "seat_count",
        "ticket_price",
        "total_amount",
        "payment_method",
        "route_id",
        "vehicle_type",
        "total_seats",
        "distance_km",
        "base_fare",
        "lead_time_days",
        "booking_month",
        "travel_month",
        "booking_day_of_week",
        "travel_day_of_week",
        "booking_hour",
        "travel_hour",
        "price_per_seat",
        "distance_category",
        "is_cancelled"
    ]

    feature_columns = [
        column
        for column in feature_columns
        if column in data.columns
    ]

    ml_data = data[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    ml_data = ml_data.dropna(
        subset=[
            "is_cancelled"
        ]
    )

    return ml_data



# DATA QUALITY CHECK


def validate_ml_dataset(
    data
):

    print("\n")
    
    print("ML DATASET VALIDATION")
    

    print(
        f"\nRows    : {len(data):,}"
    )

    print(
        f"Columns : {len(data.columns)}"
    )

    print("\nColumns:")
    print(
        data.columns.tolist()
    )

    print("\nMissing values:")

    print(
        data.isnull()
        .sum()
        .to_string()
    )

    print("\nTarget distribution:")

    target_counts = (
        data["is_cancelled"]
        .value_counts()
        .sort_index()
    )

    print(
        target_counts.to_string()
    )

    print("\nTarget percentage:")

    target_percentage = (
        data["is_cancelled"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        target_percentage
        .round(2)
        .to_string()
    )



# SAVE


def save_ml_dataset(
    data
):

    output_path = os.path.join(
        ML_DIR,
        "cancellation_ml_dataset.csv"
    )

    data.to_csv(
        output_path,
        index=False
    )

    print("\n")
    
    print("ML DATASET SAVED")
    

    print(output_path)



# MAIN


def main():

    
    print("STEP 10.1 — CANCELLATION PREDICTION DATASET")
    

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    bookings, trips, routes = load_data()

    print(
        f"\nBookings : {len(bookings):,}"
    )

    print(
        f"Trips    : {len(trips):,}"
    )

    print(
        f"Routes   : {len(routes):,}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    ml_data = prepare_ml_dataset(
        bookings,
        trips,
        routes
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_ml_dataset(
        ml_data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_ml_dataset(
        ml_data
    )

    print("\n")
    
    print("STEP 10.1 COMPLETED")
    


if __name__ == "__main__":
    main()