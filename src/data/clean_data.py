import os
import pandas as pd



BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)




def load_data():

    
    print("TICKET BOOKING ANALYTICS")
    print("DATA CLEANING & PROCESSING")
    

    customers = pd.read_csv(
        os.path.join(RAW_DIR, "customers.csv")
    )

    operators = pd.read_csv(
        os.path.join(RAW_DIR, "operators.csv")
    )

    routes = pd.read_csv(
        os.path.join(RAW_DIR, "routes.csv")
    )

    trips = pd.read_csv(
        os.path.join(RAW_DIR, "trips.csv")
    )

    bookings = pd.read_csv(
        os.path.join(RAW_DIR, "bookings.csv")
    )

    return (
        customers,
        operators,
        routes,
        trips,
        bookings
    )




def clean_customers(df):

    print("\nCleaning customers...")

    df = df.copy()

    # Remove completely duplicated rows

    df = df.drop_duplicates()

    # Remove duplicate customer IDs

    df = df.drop_duplicates(
        subset=["customer_id"],
        keep="first"
    )

    # Clean text columns

    df["customer_id"] = (
        df["customer_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["customer_name"] = (
        df["customer_name"]
        .astype(str)
        .str.strip()
    )

    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["city"] = (
        df["city"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Convert age to numeric

    df["age"] = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    # Convert registration date

    df["registration_date"] = pd.to_datetime(
        df["registration_date"],
        errors="coerce"
    )

    # Remove invalid ages

    df = df[
        df["age"].between(18, 100)
    ]

    # Remove rows with missing critical fields

    df = df.dropna(
        subset=[
            "customer_id",
            "customer_name",
            "registration_date"
        ]
    )

    return df




def clean_operators(df):

    print("Cleaning operators...")

    df = df.copy()

    df = df.drop_duplicates()

    df = df.drop_duplicates(
        subset=["operator_id"],
        keep="first"
    )

    df["operator_id"] = (
        df["operator_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["operator_name"] = (
        df["operator_name"]
        .astype(str)
        .str.strip()
    )

    df["city"] = (
        df["city"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df = df.dropna(
        subset=[
            "operator_id",
            "operator_name"
        ]
    )

    return df




def clean_routes(df):

    print("Cleaning routes...")

    df = df.copy()

    df = df.drop_duplicates()

    df = df.drop_duplicates(
        subset=["route_id"],
        keep="first"
    )

    # Clean IDs

    df["route_id"] = (
        df["route_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Clean text

    for column in [
        "route_name",
        "source",
        "destination"
    ]:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Numeric columns

    df["distance_km"] = pd.to_numeric(
        df["distance_km"],
        errors="coerce"
    )

    df["base_fare"] = pd.to_numeric(
        df["base_fare"],
        errors="coerce"
    )

    # Remove invalid values

    df = df[
        (df["distance_km"] > 0) &
        (df["base_fare"] > 0)
    ]

    # Source and destination cannot be same

    df = df[
        df["source"] != df["destination"]
    ]

    df = df.dropna(
        subset=[
            "route_id",
            "source",
            "destination"
        ]
    )

    return df




def clean_trips(df):

    print("Cleaning trips...")

    df = df.copy()

    df = df.drop_duplicates()

    df = df.drop_duplicates(
        subset=["trip_id"],
        keep="first"
    )

    # IDs

    df["trip_id"] = (
        df["trip_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["route_id"] = (
        df["route_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["operator_id"] = (
        df["operator_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Vehicle type

    df["vehicle_type"] = (
        df["vehicle_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Datetime

    df["departure_datetime"] = pd.to_datetime(
        df["departure_datetime"],
        errors="coerce"
    )

    df["arrival_datetime"] = pd.to_datetime(
        df["arrival_datetime"],
        errors="coerce"
    )

    # Seats

    df["total_seats"] = pd.to_numeric(
        df["total_seats"],
        errors="coerce"
    )

    # Remove invalid rows

    df = df[
        df["total_seats"] > 0
    ]

    df = df[
        df["arrival_datetime"] >
        df["departure_datetime"]
    ]

    df = df.dropna(
        subset=[
            "trip_id",
            "route_id",
            "operator_id",
            "departure_datetime",
            "arrival_datetime"
        ]
    )

    return df




def clean_bookings(df):

    print("Cleaning bookings...")

    df = df.copy()

    df = df.drop_duplicates()

    df = df.drop_duplicates(
        subset=["booking_id"],
        keep="first"
    )

    # IDs

    for column in [
        "booking_id",
        "customer_id",
        "trip_id"
    ]:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Dates

    df["booking_date"] = pd.to_datetime(
        df["booking_date"],
        errors="coerce"
    )

    df["travel_date"] = pd.to_datetime(
        df["travel_date"],
        errors="coerce"
    )

    # Numeric fields

    df["seat_count"] = pd.to_numeric(
        df["seat_count"],
        errors="coerce"
    )

    df["ticket_price"] = pd.to_numeric(
        df["ticket_price"],
        errors="coerce"
    )

    df["total_amount"] = pd.to_numeric(
        df["total_amount"],
        errors="coerce"
    )

    # Payment method

    df["payment_method"] = (
        df["payment_method"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Booking status

    df["booking_status"] = (
        df["booking_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Valid values

    valid_payment_methods = [
        "UPI",
        "CARD",
        "NETBANKING",
        "WALLET",
        "CASH"
    ]

    valid_statuses = [
        "CONFIRMED",
        "CANCELLED",
        "PENDING"
    ]

    df = df[
        df["payment_method"].isin(
            valid_payment_methods
        )
    ]

    df = df[
        df["booking_status"].isin(
            valid_statuses
        )
    ]

    # Valid numeric values

    df = df[
        df["seat_count"] > 0
    ]

    df = df[
        df["ticket_price"] > 0
    ]

    df = df[
        df["total_amount"] > 0
    ]

    # Booking cannot happen after travel

    df = df[
        df["booking_date"] <=
        df["travel_date"]
    ]

    # Remove rows missing critical fields

    df = df.dropna(
        subset=[
            "booking_id",
            "customer_id",
            "trip_id",
            "booking_date",
            "travel_date"
        ]
    )

    return df



def validate_relationships(
    customers,
    operators,
    routes,
    trips,
    bookings
):

    print("\nChecking dataset relationships...")

    # Trips → Routes

    valid_route_ids = set(
        routes["route_id"]
    )

    trips = trips[
        trips["route_id"].isin(
            valid_route_ids
        )
    ]

    # Trips → Operators

    valid_operator_ids = set(
        operators["operator_id"]
    )

    trips = trips[
        trips["operator_id"].isin(
            valid_operator_ids
        )
    ]

    # Bookings → Customers

    valid_customer_ids = set(
        customers["customer_id"]
    )

    bookings = bookings[
        bookings["customer_id"].isin(
            valid_customer_ids
        )
    ]

    # Bookings → Trips

    valid_trip_ids = set(
        trips["trip_id"]
    )

    bookings = bookings[
        bookings["trip_id"].isin(
            valid_trip_ids
        )
    ]

    return (
        trips,
        bookings
    )



# SAVE DATA


def save_data(
    customers,
    operators,
    routes,
    trips,
    bookings
):

    print("\nSaving processed datasets...")

    customers.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "customers_clean.csv"
        ),
        index=False
    )

    operators.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "operators_clean.csv"
        ),
        index=False
    )

    routes.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "routes_clean.csv"
        ),
        index=False
    )

    trips.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "trips_clean.csv"
        ),
        index=False
    )

    bookings.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "bookings_clean.csv"
        ),
        index=False
    )



def print_summary(
    customers,
    operators,
    routes,
    trips,
    bookings
):

    print("\n")
    
    print("CLEANING SUMMARY")
    

    print(
        f"Customers : {len(customers):,}"
    )

    print(
        f"Operators : {len(operators):,}"
    )

    print(
        f"Routes    : {len(routes):,}"
    )

    print(
        f"Trips     : {len(trips):,}"
    )

    print(
        f"Bookings  : {len(bookings):,}"
    )

    print("\nProcessed files saved successfully.")

    print(
        f"\nLocation: {PROCESSED_DIR}"
    )




def main():

    (
        customers,
        operators,
        routes,
        trips,
        bookings
    ) = load_data()

    # Clean each dataset

    customers = clean_customers(
        customers
    )

    operators = clean_operators(
        operators
    )

    routes = clean_routes(
        routes
    )

    trips = clean_trips(
        trips
    )

    bookings = clean_bookings(
        bookings
    )

    # Check relationships

    (
        trips,
        bookings
    ) = validate_relationships(
        customers,
        operators,
        routes,
        trips,
        bookings
    )

    # Save

    save_data(
        customers,
        operators,
        routes,
        trips,
        bookings
    )

    # Summary

    print_summary(
        customers,
        operators,
        routes,
        trips,
        bookings
    )



# RUN


if __name__ == "__main__":
    main()
