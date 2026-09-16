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
        "departure_datetime",
        "total_seats"
    ]

    trip_columns = [
        c for c in trip_columns
        if c in trips.columns
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
        c for c in route_columns
        if c in routes.columns
    ]

    data = data.merge(
        routes[route_columns],
        on="route_id",
        how="left"
    )

    # --------------------------------------------------------
    # Normalize status
    # --------------------------------------------------------

    data["booking_status"] = (
        data["booking_status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Confirmed bookings
    # --------------------------------------------------------

    data["is_confirmed"] = (
        data["booking_status"]
        == "CONFIRMED"
    )

    # --------------------------------------------------------
    # Date attributes
    # --------------------------------------------------------

    data["booking_year"] = (
        data["booking_date"].dt.year
    )

    data["booking_month_number"] = (
        data["booking_date"].dt.month
    )

    data["booking_month"] = (
        data["booking_date"]
        .dt.to_period("M")
        .astype(str)
    )

    data["booking_month_name"] = (
        data["booking_date"]
        .dt.month_name()
    )

    data["booking_day"] = (
        data["booking_date"].dt.day
    )

    data["booking_day_of_week"] = (
        data["booking_date"].dt.day_name()
    )

    data["booking_week"] = (
        data["booking_date"]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )

    # --------------------------------------------------------
    # Travel date attributes
    # --------------------------------------------------------

    data["travel_year"] = (
        data["travel_date"].dt.year
    )

    data["travel_month_number"] = (
        data["travel_date"].dt.month
    )

    data["travel_month"] = (
        data["travel_date"]
        .dt.to_period("M")
        .astype(str)
    )

    data["travel_month_name"] = (
        data["travel_date"]
        .dt.month_name()
    )

    data["travel_day_of_week"] = (
        data["travel_date"].dt.day_name()
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    def get_season(month):

        if month in [12, 1, 2]:
            return "Winter"

        elif month in [3, 4, 5]:
            return "Summer"

        elif month in [6, 7, 8, 9]:
            return "Monsoon"

        else:
            return "Autumn"

    data["travel_season"] = (
        data["travel_month_number"]
        .apply(get_season)
    )

    return data



# MONTHLY BOOKING DEMAND


def monthly_booking_demand(data):

    result = (
        data
        .groupby("booking_month")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
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

    result["confirmation_rate"] = (
        result["confirmed_bookings"]
        /
        result["bookings"]
        *
        100
    )

    result["booking_growth_percentage"] = (
        result["bookings"]
        .pct_change()
        *
        100
    )

    return result



# MONTHLY TRAVEL DEMAND


def monthly_travel_demand(data):

    result = (
        data
        .groupby("travel_month")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
                "sum"
            )
        )
        .reset_index()
    )

    result["confirmation_rate"] = (
        result["confirmed_bookings"]
        /
        result["bookings"]
        *
        100
    )

    return result



# DAY OF WEEK DEMAND


def weekday_demand(data):

    order = [
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
        .groupby("travel_day_of_week")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
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

    result["day_order"] = (
        result["travel_day_of_week"]
        .map(
            {
                day: i
                for i, day in enumerate(order)
            }
        )
    )

    return (
        result
        .sort_values("day_order")
        .drop(columns="day_order")
    )



# MONTH OF YEAR SEASONALITY


def monthly_seasonality(data):

    result = (
        data
        .groupby(
            [
                "travel_month_number",
                "travel_month_name"
            ]
        )
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
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

    return result.sort_values(
        "travel_month_number"
    )



# SEASON ANALYSIS


def season_analysis(data):

    result = (
        data
        .groupby("travel_season")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
                "seat_count",
                "sum"
            ),

            revenue=(
                "total_amount",
                "sum"
            ),

            confirmed_bookings=(
                "is_confirmed",
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

    result["demand_share"] = (
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



# YEARLY DEMAND


def yearly_demand(data):

    result = (
        data
        .groupby("travel_year")
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
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

    result["booking_growth_percentage"] = (
        result["bookings"]
        .pct_change()
        *
        100
    )

    result["revenue_growth_percentage"] = (
        result["revenue"]
        .pct_change()
        *
        100
    )

    return result



# ROUTE SEASONAL DEMAND


def route_seasonality(data):

    result = (
        data
        .groupby(
            [
                "route_id",
                "route_name",
                "source",
                "destination",
                "travel_season"
            ]
        )
        .agg(
            bookings=(
                "booking_id",
                "count"
            ),

            tickets=(
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
        "bookings",
        ascending=False
    )



# PEAK / OFF-PEAK CLASSIFICATION


def classify_peak_periods(monthly):

    result = monthly.copy()

    average_demand = (
        result["bookings"].mean()
    )

    result["demand_period"] = (
        result["bookings"]
        .apply(
            lambda x:
            "Peak"
            if x >= average_demand * 1.20
            else (
                "Off-Peak"
                if x <= average_demand * 0.80
                else "Normal"
            )
        )
    )

    return result



# PEAK MONTHS


def get_peak_months(monthly):

    return (
        monthly
        .sort_values(
            "bookings",
            ascending=False
        )
        .head(5)
        .copy()
    )



# LOW DEMAND MONTHS


def get_low_demand_months(monthly):

    return (
        monthly
        .sort_values(
            "bookings",
            ascending=True
        )
        .head(5)
        .copy()
    )



# BUSINESS INSIGHTS


def generate_insights(
    monthly,
    travel_monthly,
    weekday,
    season,
    yearly,
    peak_months,
    low_months
):

    print("\n")
    
    print("SEASONAL & TIME-SERIES DEMAND INSIGHTS")
    

    # --------------------------------------------------------
    # Peak months
    # --------------------------------------------------------

    print("\n1. TOP 5 PEAK BOOKING MONTHS")
    print("-" * 80)

    print(
        peak_months[
            [
                "booking_month",
                "bookings",
                "tickets",
                "revenue"
            ]
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Low months
    # --------------------------------------------------------

    print("\n2. TOP 5 LOW-DEMAND BOOKING MONTHS")
    print("-" * 80)

    print(
        low_months[
            [
                "booking_month",
                "bookings",
                "tickets",
                "revenue"
            ]
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    print("\n3. SEASONAL DEMAND")
    print("-" * 80)

    print(
        season.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    print("\n4. DEMAND BY TRAVEL DAY")
    print("-" * 80)

    print(
        weekday.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Yearly
    # --------------------------------------------------------

    print("\n5. YEARLY DEMAND")
    print("-" * 80)

    print(
        yearly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    print("\n6. BUSINESS RECOMMENDATIONS")
    print("-" * 80)

    if not peak_months.empty:

        peak = peak_months.iloc[0]

        print(
            f"• Highest booking demand was in "
            f"{peak['booking_month']} "
            f"with {peak['bookings']:,} bookings."
        )

    if not low_months.empty:

        low = low_months.iloc[0]

        print(
            f"• Lowest booking demand was in "
            f"{low['booking_month']} "
            f"with {low['bookings']:,} bookings."
        )

    if not season.empty:

        peak_season = season.iloc[0]

        print(
            f"• {peak_season['travel_season']} "
            f"was the strongest season based on "
            f"booking demand."
        )

    if not weekday.empty:

        busiest_day = weekday.loc[
            weekday["bookings"].idxmax()
        ]

        print(
            f"• {busiest_day['travel_day_of_week']} "
            f"had the highest travel demand."
        )

    print(
        "• Increase capacity during consistently "
        "high-demand periods."
    )

    print(
        "• Use targeted promotions during "
        "low-demand periods."
    )

    print(
        "• Use historical seasonality as an input "
        "for future demand forecasting."
    )



# SAVE RESULTS


def save_results(
    monthly,
    travel_monthly,
    weekday,
    season,
    monthly_season,
    yearly,
    route_season,
    peak_months,
    low_months
):

    results = {

        "monthly_booking_demand.csv":
            monthly,

        "monthly_travel_demand.csv":
            travel_monthly,

        "weekday_demand.csv":
            weekday,

        "season_demand.csv":
            season,

        "monthly_seasonality.csv":
            monthly_season,

        "yearly_demand.csv":
            yearly,

        "route_seasonality.csv":
            route_season,

        "peak_booking_months.csv":
            peak_months,

        "low_demand_months.csv":
            low_months
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

    
    print("STEP 9.9 — SEASONAL & TIME-SERIES DEMAND ANALYTICS")
    

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
    # Analytics
    # --------------------------------------------------------

    monthly = monthly_booking_demand(
        data
    )

    travel_monthly = monthly_travel_demand(
        data
    )

    weekday = weekday_demand(
        data
    )

    monthly_season = monthly_seasonality(
        data
    )

    season = season_analysis(
        data
    )

    yearly = yearly_demand(
        data
    )

    route_season = route_seasonality(
        data
    )

    # --------------------------------------------------------
    # Peak classification
    # --------------------------------------------------------

    monthly = classify_peak_periods(
        monthly
    )

    peak_months = get_peak_months(
        monthly
    )

    low_months = get_low_demand_months(
        monthly
    )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_insights(
        monthly,
        travel_monthly,
        weekday,
        season,
        yearly,
        peak_months,
        low_months
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        monthly,
        travel_monthly,
        weekday,
        season,
        monthly_season,
        yearly,
        route_season,
        peak_months,
        low_months
    )

    print("\n")
    
    print("STEP 9.9 COMPLETED")
    


if __name__ == "__main__":
    main()