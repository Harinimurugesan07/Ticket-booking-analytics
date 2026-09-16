import os
import pandas as pd




BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)



def load_datasets():

    
    print("TICKET BOOKING ANALYTICS")
    print("DATA VALIDATION")
    

    customers = pd.read_csv(
        os.path.join(DATA_DIR, "customers.csv")
    )

    operators = pd.read_csv(
        os.path.join(DATA_DIR, "operators.csv")
    )

    routes = pd.read_csv(
        os.path.join(DATA_DIR, "routes.csv")
    )

    trips = pd.read_csv(
        os.path.join(DATA_DIR, "trips.csv")
    )

    bookings = pd.read_csv(
        os.path.join(DATA_DIR, "bookings.csv")
    )

    return {
        "customers": customers,
        "operators": operators,
        "routes": routes,
        "trips": trips,
        "bookings": bookings
    }



def validate_basic_info(name, df):

    print("\n")
    
    print(f"DATASET: {name.upper()}")
    

    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns)}")
    print(f"Duplicate Rows    : {df.duplicated().sum():,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing Values:")

    missing = df.isnull().sum()

    for column, count in missing.items():

        print(
            f"  {column:<25} : {count:,}"
        )

    print("\nData Types:")

    for column, dtype in df.dtypes.items():

        print(
            f"  {column:<25} : {dtype}"
        )




def validate_customers(df):

    print("\n")
    print("-" * 80)
    print("CUSTOMERS VALIDATION")
    print("-" * 80)

    print(
        "Customer ID duplicates:",
        df["customer_id"].duplicated().sum()
    )

    print(
        "Invalid ages:",
        (
            (df["age"] < 18) |
            (df["age"] > 100)
        ).sum()
    )

    print(
        "Invalid genders:",
        (~df["gender"].isin(
            ["Male", "Female", "Other"]
        )).sum()
    )



def validate_operators(df):

    print("\n")
    print("-" * 80)
    print("OPERATORS VALIDATION")
    print("-" * 80)

    print(
        "Operator ID duplicates:",
        df["operator_id"].duplicated().sum()
    )

    print(
        "Empty operator names:",
        df["operator_name"].isna().sum()
    )




def validate_routes(df):

    print("\n")
    print("-" * 80)
    print("ROUTES VALIDATION")
    print("-" * 80)

    print(
        "Route ID duplicates:",
        df["route_id"].duplicated().sum()
    )

    print(
        "Invalid distance:",
        (df["distance_km"] <= 0).sum()
    )

    print(
        "Invalid base fare:",
        (df["base_fare"] <= 0).sum()
    )

    print(
        "Source = Destination:",
        (
            df["source"] ==
            df["destination"]
        ).sum()
    )




def validate_trips(df, routes, operators):

    print("\n")
    print("-" * 80)
    print("TRIPS VALIDATION")
    print("-" * 80)

    print(
        "Trip ID duplicates:",
        df["trip_id"].duplicated().sum()
    )

    # Check route references

    invalid_routes = (
        ~df["route_id"].isin(
            routes["route_id"]
        )
    ).sum()

    print(
        "Invalid route references:",
        invalid_routes
    )

    # Check operator references

    invalid_operators = (
        ~df["operator_id"].isin(
            operators["operator_id"]
        )
    ).sum()

    print(
        "Invalid operator references:",
        invalid_operators
    )

    # Vehicle validation

    valid_vehicle_types = [
        "BUS",
        "CAB",
        "BIKE",
        "AUTO"
    ]

    invalid_vehicle_types = (
        ~df["vehicle_type"].isin(
            valid_vehicle_types
        )
    ).sum()

    print(
        "Invalid vehicle types:",
        invalid_vehicle_types
    )

    # Seat validation

    print(
        "Invalid seat capacity:",
        (df["total_seats"] <= 0).sum()
    )




def validate_bookings(
    df,
    customers,
    trips
):

    print("\n")
    print("-" * 80)
    print("BOOKINGS VALIDATION")
    print("-" * 80)

    print(
        "Booking ID duplicates:",
        df["booking_id"].duplicated().sum()
    )

    # Customer references

    invalid_customers = (
        ~df["customer_id"].isin(
            customers["customer_id"]
        )
    ).sum()

    print(
        "Invalid customer references:",
        invalid_customers
    )

    # Trip references

    invalid_trips = (
        ~df["trip_id"].isin(
            trips["trip_id"]
        )
    ).sum()

    print(
        "Invalid trip references:",
        invalid_trips
    )

    # Seat count

    print(
        "Invalid seat count:",
        (df["seat_count"] <= 0).sum()
    )

    # Ticket price

    print(
        "Invalid ticket price:",
        (df["ticket_price"] <= 0).sum()
    )

    # Total amount

    print(
        "Invalid total amount:",
        (df["total_amount"] <= 0).sum()
    )

    # Payment methods

    valid_payment_methods = [
        "UPI",
        "CARD",
        "NETBANKING",
        "WALLET",
        "CASH"
    ]

    invalid_payment_methods = (
        ~df["payment_method"].isin(
            valid_payment_methods
        )
    ).sum()

    print(
        "Invalid payment methods:",
        invalid_payment_methods
    )

    # Booking statuses

    valid_statuses = [
        "CONFIRMED",
        "CANCELLED",
        "PENDING"
    ]

    invalid_statuses = (
        ~df["booking_status"].isin(
            valid_statuses
        )
    ).sum()

    print(
        "Invalid booking statuses:",
        invalid_statuses
    )




def validate_dates(
    customers,
    trips,
    bookings
):

    print("\n")
    print("-" * 80)
    print("DATE VALIDATION")
    print("-" * 80)

    # Customers

    customer_dates = pd.to_datetime(
        customers["registration_date"],
        errors="coerce"
    )

    print(
        "Invalid customer registration dates:",
        customer_dates.isna().sum()
    )

    # Trips

    departure_dates = pd.to_datetime(
        trips["departure_datetime"],
        errors="coerce"
    )

    arrival_dates = pd.to_datetime(
        trips["arrival_datetime"],
        errors="coerce"
    )

    print(
        "Invalid departure dates:",
        departure_dates.isna().sum()
    )

    print(
        "Invalid arrival dates:",
        arrival_dates.isna().sum()
    )

    print(
        "Arrival before departure:",
        (arrival_dates < departure_dates).sum()
    )

    # Bookings

    booking_dates = pd.to_datetime(
        bookings["booking_date"],
        errors="coerce"
    )

    travel_dates = pd.to_datetime(
        bookings["travel_date"],
        errors="coerce"
    )

    print(
        "Invalid booking dates:",
        booking_dates.isna().sum()
    )

    print(
        "Invalid travel dates:",
        travel_dates.isna().sum()
    )

    print(
        "Booking after travel:",
        (booking_dates > travel_dates).sum()
    )




def main():

    datasets = load_datasets()

    customers = datasets["customers"]
    operators = datasets["operators"]
    routes = datasets["routes"]
    trips = datasets["trips"]
    bookings = datasets["bookings"]

    # Basic information

    for name, df in datasets.items():

        validate_basic_info(
            name,
            df
        )

    # Dataset-specific validation

    validate_customers(
        customers
    )

    validate_operators(
        operators
    )

    validate_routes(
        routes
    )

    validate_trips(
        trips,
        routes,
        operators
    )

    validate_bookings(
        bookings,
        customers,
        trips
    )

    validate_dates(
        customers,
        trips,
        bookings
    )

    # Final message

    print("\n")
    
    print("DATA VALIDATION COMPLETED")
    




if __name__ == "__main__":
    main()
