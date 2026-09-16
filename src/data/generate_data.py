import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd




RANDOM_SEED = 42

NUM_CUSTOMERS = 5000
NUM_OPERATORS = 20
NUM_ROUTES = 30
NUM_TRIPS = 300
NUM_BOOKINGS = 20000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)




CITIES = [
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Trichy",
    "Salem",
    "Thanjavur",
    "Tirunelveli",
    "Erode",
    "Vellore",
    "Bengaluru",
    "Hyderabad",
    "Pondicherry",
]

VEHICLE_TYPES = [
    "BUS",
    "CAB",
    "BIKE",
    "AUTO",
]

PAYMENT_METHODS = [
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET",
    "CASH",
]

BOOKING_STATUSES = [
    "CONFIRMED",
    "CANCELLED",
    "PENDING",
]




def generate_customers():

    customers = []

    start_date = datetime(2024, 1, 1)

    for i in range(1, NUM_CUSTOMERS + 1):

        registration_date = start_date + timedelta(
            days=random.randint(0, 900)
        )

        customers.append({
            "customer_id": f"CUST{i:05d}",
            "customer_name": f"Customer {i}",
            "gender": random.choice(["Male", "Female", "Other"]),
            "age": random.randint(18, 65),
            "city": random.choice(CITIES),
            "registration_date": registration_date.date(),
        })

    return pd.DataFrame(customers)




def generate_operators():

    operators = []

    for i in range(1, NUM_OPERATORS + 1):

        operators.append({
            "operator_id": f"OP{i:03d}",
            "operator_name": f"Travel Operator {i}",
            "city": random.choice(CITIES),
        })

    return pd.DataFrame(operators)




def generate_routes():

    routes = []

    route_id = 1

    possible_routes = []

    for source in CITIES:
        for destination in CITIES:

            if source != destination:
                possible_routes.append(
                    (source, destination)
                )

    random.shuffle(possible_routes)

    selected_routes = possible_routes[:NUM_ROUTES]

    for source, destination in selected_routes:

        distance = random.randint(50, 650)

        base_fare = round(
            100 + (distance * random.uniform(1.5, 3.0)),
            2
        )

        routes.append({
            "route_id": f"R{route_id:03d}",
            "route_name": f"{source} to {destination}",
            "source": source,
            "destination": destination,
            "distance_km": distance,
            "base_fare": base_fare,
        })

        route_id += 1

    return pd.DataFrame(routes)



def generate_trips(routes, operators):

    trips = []

    start_date = datetime(2025, 1, 1)

    for i in range(1, NUM_TRIPS + 1):

        route = routes.sample(1).iloc[0]
        operator = operators.sample(1).iloc[0]

        departure = start_date + timedelta(
            days=random.randint(0, 730),
            hours=random.randint(5, 22),
            minutes=random.choice([0, 15, 30, 45])
        )

        duration_hours = max(
            1,
            route["distance_km"] / random.uniform(45, 70)
        )

        arrival = departure + timedelta(
            hours=duration_hours
        )

        vehicle_type = random.choice(
            VEHICLE_TYPES
        )

        if vehicle_type == "BUS":
            total_seats = random.choice(
                [30, 40, 45, 50]
            )

        elif vehicle_type == "CAB":
            total_seats = random.choice(
                [4, 5, 6]
            )

        elif vehicle_type == "AUTO":
            total_seats = 3

        else:
            total_seats = 1

        trips.append({
            "trip_id": f"TRIP{i:05d}",
            "route_id": route["route_id"],
            "operator_id": operator["operator_id"],
            "vehicle_type": vehicle_type,
            "departure_datetime": departure,
            "arrival_datetime": arrival,
            "total_seats": total_seats,
        })

    return pd.DataFrame(trips)




def generate_bookings(customers, trips, routes):

    bookings = []

    booking_start = datetime(2025, 1, 1)

    for i in range(1, NUM_BOOKINGS + 1):

        customer = customers.sample(1).iloc[0]
        trip = trips.sample(1).iloc[0]

        route = routes[
            routes["route_id"] == trip["route_id"]
        ].iloc[0]

        travel_date = pd.to_datetime(
            trip["departure_datetime"]
        )

        # Booking should happen before travel.
        days_before = random.randint(1, 60)

        booking_date = (
            travel_date -
            timedelta(days=days_before)
        )

        # Number of seats/tickets
        if trip["vehicle_type"] == "BUS":
            seat_count = random.randint(1, 5)

        elif trip["vehicle_type"] == "CAB":
            seat_count = random.randint(1, 4)

        elif trip["vehicle_type"] == "AUTO":
            seat_count = random.randint(1, 3)

        else:
            seat_count = 1

        # Small price variation
        ticket_price = round(
            route["base_fare"] *
            random.uniform(0.90, 1.20),
            2
        )

        total_amount = round(
            ticket_price * seat_count,
            2
        )

        # Booking status
        status_probability = random.random()

        if status_probability < 0.82:
            booking_status = "CONFIRMED"

        elif status_probability < 0.95:
            booking_status = "CANCELLED"

        else:
            booking_status = "PENDING"

        bookings.append({
            "booking_id": f"BOOK{i:06d}",
            "customer_id": customer["customer_id"],
            "trip_id": trip["trip_id"],
            "booking_date": booking_date.date(),
            "travel_date": travel_date.date(),
            "seat_count": seat_count,
            "ticket_price": ticket_price,
            "total_amount": total_amount,
            "payment_method": random.choice(
                PAYMENT_METHODS
            ),
            "booking_status": booking_status,
        })

    return pd.DataFrame(bookings)




def save_data():

    print("=" * 70)
    print("TICKET BOOKING ANALYTICS - DATA GENERATION")
    print("=" * 70)

    print("\nGenerating customers...")
    customers = generate_customers()

    print("Generating operators...")
    operators = generate_operators()

    print("Generating routes...")
    routes = generate_routes()

    print("Generating trips...")
    trips = generate_trips(
        routes,
        operators
    )

    print("Generating bookings...")
    bookings = generate_bookings(
        customers,
        trips,
        routes
    )

    # Save CSV files

    customers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "customers.csv"
        ),
        index=False
    )

    operators.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "operators.csv"
        ),
        index=False
    )

    routes.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "routes.csv"
        ),
        index=False
    )

    trips.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "trips.csv"
        ),
        index=False
    )

    bookings.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "bookings.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETED")
    print("=" * 70)

    print(f"\nCustomers : {len(customers):,}")
    print(f"Operators : {len(operators):,}")
    print(f"Routes    : {len(routes):,}")
    print(f"Trips     : {len(trips):,}")
    print(f"Bookings  : {len(bookings):,}")

    print(f"\nFiles saved to:")
    print(OUTPUT_DIR)




if __name__ == "__main__":
    save_data()

