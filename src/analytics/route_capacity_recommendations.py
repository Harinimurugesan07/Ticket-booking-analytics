import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 11.4 — ROUTE & CAPACITY RECOMMENDATIONS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

BOOKINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bookings_clean.csv"
)

TRIPS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "trips_clean.csv"
)

ROUTES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "routes_clean.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "recommendations"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "route_capacity_recommendations.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    bookings = pd.read_csv(
        BOOKINGS_PATH
    )

    trips = pd.read_csv(
        TRIPS_PATH
    )

    routes = pd.read_csv(
        ROUTES_PATH
    )

    bookings = bookings[
        bookings[
            "booking_status"
        ]
        .astype(str)
        .str.upper()
        .eq("CONFIRMED")
    ].copy()

    bookings["seat_count"] = pd.to_numeric(
        bookings["seat_count"],
        errors="coerce"
    ).fillna(0)

    return (
        bookings,
        trips,
        routes
    )


# ============================================================
# ROUTE ANALYTICS
# ============================================================

def calculate_route_metrics(
    bookings,
    trips,
    routes
):

    data = bookings.merge(
        trips[
            [
                "trip_id",
                "route_id",
                "total_seats",
                "vehicle_type"
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

    route_metrics = (
        data.groupby(
            [
                "route_id",
                "route_name",
                "source",
                "destination"
            ],
            dropna=False
        )
        .agg(
            booking_count=(
                "booking_id",
                "count"
            ),
            seats_booked=(
                "seat_count",
                "sum"
            ),
            total_revenue=(
                "total_amount",
                "sum"
            ),
            average_distance_km=(
                "distance_km",
                "mean"
            )
        )
        .reset_index()
    )

    return route_metrics


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    route_metrics
):

    if route_metrics.empty:

        return pd.DataFrame()

    metrics = route_metrics.copy()

    booking_threshold = (
        metrics["booking_count"]
        .quantile(0.75)
    )

    low_booking_threshold = (
        metrics["booking_count"]
        .quantile(0.25)
    )

    recommendations = []

    for _, row in metrics.iterrows():

        bookings = row[
            "booking_count"
        ]

        route = row[
            "route_id"
        ]

        route_name = row[
            "route_name"
        ]

        if bookings >= booking_threshold:

            priority = "HIGH"

            recommendation_type = (
                "Route Capacity Expansion"
            )

            recommendation = (
                "Increase trip frequency or "
                "seat capacity on this route."
            )

            reason = (
                "Route booking demand is in the "
                "top demand quartile."
            )

        elif bookings <= low_booking_threshold:

            priority = "MEDIUM"

            recommendation_type = (
                "Route Demand Stimulation"
            )

            recommendation = (
                "Use targeted promotions and "
                "review trip frequency."
            )

            reason = (
                "Route booking demand is in the "
                "lowest demand quartile."
            )

        else:

            priority = "LOW"

            recommendation_type = (
                "Route Monitoring"
            )

            recommendation = (
                "Continue monitoring booking trends "
                "and maintain current capacity."
            )

            reason = (
                "Route demand is within the normal "
                "range."
            )

        recommendations.append(
            {
                "recommendation_type":
                    recommendation_type,

                "entity_type":
                    "Route",

                "entity_id":
                    route,

                "route_name":
                    route_name,

                "source":
                    row["source"],

                "destination":
                    row["destination"],

                "booking_count":
                    int(bookings),

                "seats_booked":
                    row["seats_booked"],

                "total_revenue":
                    round(
                        row["total_revenue"],
                        2
                    ),

                "priority":
                    priority,

                "recommendation":
                    recommendation,

                "business_reason":
                    reason
            }
        )

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "STEP 11.4 — ROUTE & CAPACITY RECOMMENDATIONS"
    )
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    bookings, trips, routes = load_data()

    print(
        f"\nConfirmed bookings: "
        f"{len(bookings):,}"
    )

    print(
        f"Routes: "
        f"{len(routes):,}"
    )

    metrics = calculate_route_metrics(
        bookings,
        trips,
        routes
    )

    result = generate_recommendations(
        metrics
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print(
        result[
            "recommendation_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\nOutput saved:")

    print(
        OUTPUT_PATH
    )

    print("\n")
    print("=" * 80)
    print(
        "STEP 11.4 COMPLETED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()