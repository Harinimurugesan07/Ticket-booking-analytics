import os
import joblib
import pandas as pd

from datetime import datetime



# STEP 10.8
# NEW BOOKING CANCELLATION PREDICTION




# PROJECT PATHS


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ml",
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "ml"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression.pkl"
)



# CONSTANTS


PAYMENT_METHODS = [
    "CARD",
    "CASH",
    "NETBANKING",
    "UPI",
    "WALLET"
]

VEHICLE_TYPES = [
    "AUTO",
    "BIKE",
    "BUS",
    "CAB"
]

ROUTE_IDS = [
    f"R{i:03d}"
    for i in range(1, 31)
]

DISTANCE_CATEGORIES = [
    "Long",
    "Medium",
    "Very Long"
]



# LOAD MODEL


def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nLogistic Regression model not found:\n"
            f"{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "\nModel loaded successfully."
    )

    return model



# VALIDATION HELPERS


def get_integer(
    message,
    minimum=0
):

    while True:

        try:

            value = int(
                input(message)
            )

            if value < minimum:

                print(
                    f"Please enter a value >= {minimum}."
                )

                continue

            return value

        except ValueError:

            print(
                "Please enter a valid integer."
            )


def get_float(
    message,
    minimum=0
):

    while True:

        try:

            value = float(
                input(message)
            )

            if value < minimum:

                print(
                    f"Please enter a value >= {minimum}."
                )

                continue

            return value

        except ValueError:

            print(
                "Please enter a valid number."
            )


def get_choice(
    message,
    choices
):

    while True:

        value = input(
            message
        ).strip().upper()

        # Vehicle/payment values are uppercase
        # but distance categories need special handling.

        for choice in choices:

            if value == choice.upper():

                return choice

        print(
            f"Invalid value."
        )

        print(
            "Available options:"
        )

        print(
            ", ".join(choices)
        )


def get_date(
    message
):

    while True:

        value = input(
            message
        ).strip()

        try:

            return datetime.strptime(
                value,
                "%Y-%m-%d"
            )

        except ValueError:

            print(
                "Invalid date."
            )

            print(
                "Use format: YYYY-MM-DD"
            )



# DISTANCE CATEGORY


def calculate_distance_category(
    distance_km
):

    if distance_km < 200:

        return "Medium"

    elif distance_km < 350:

        return "Long"

    else:

        return "Very Long"



# CREATE RAW BOOKING


def collect_booking_details():

    print("\n")
    
    print("ENTER NEW BOOKING DETAILS")
    

    # --------------------------------------------------------
    # Basic booking information
    # --------------------------------------------------------

    seat_count = get_integer(
        "\nSeat count: ",
        minimum=1
    )

    ticket_price = get_float(
        "Ticket price per seat: ",
        minimum=0
    )

    total_amount = (
        ticket_price * seat_count
    )

    print(
        f"Calculated total amount: "
        f"{total_amount:.2f}"
    )

    # --------------------------------------------------------
    # Trip information
    # --------------------------------------------------------

    total_seats = get_integer(
        "Total seats available: ",
        minimum=1
    )

    distance_km = get_float(
        "Distance (km): ",
        minimum=0
    )

    base_fare = get_float(
        "Base fare: ",
        minimum=0
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    booking_datetime = get_date(
        "Booking date (YYYY-MM-DD): "
    )

    travel_datetime = get_date(
        "Travel date (YYYY-MM-DD): "
    )

    if travel_datetime < booking_datetime:

        raise ValueError(
            "\nTravel date cannot be earlier "
            "than booking date."
        )

    # --------------------------------------------------------
    # Calculate date features
    # --------------------------------------------------------

    lead_time_days = (
        travel_datetime - booking_datetime
    ).days

    booking_month = (
        booking_datetime.month
    )

    travel_month = (
        travel_datetime.month
    )

    booking_day_of_week = (
        booking_datetime.weekday()
    )

    travel_day_of_week = (
        travel_datetime.weekday()
    )

    # Historical dataset contains booking hour
    # and travel hour.
    #
    # Since we collect dates only, ask separately.

    booking_hour = get_integer(
        "Booking hour (0-23): ",
        minimum=0
    )

    if booking_hour > 23:

        raise ValueError(
            "Booking hour must be between 0 and 23."
        )

    travel_hour = get_integer(
        "Travel hour (0-23): ",
        minimum=0
    )

    if travel_hour > 23:

        raise ValueError(
            "Travel hour must be between 0 and 23."
        )

    # --------------------------------------------------------
    # Payment
    # --------------------------------------------------------

    print(
        "\nPayment methods:"
    )

    print(
        ", ".join(PAYMENT_METHODS)
    )

    payment_method = get_choice(
        "Payment method: ",
        PAYMENT_METHODS
    )

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    print(
        "\nRoute IDs:"
    )

    print(
        ", ".join(ROUTE_IDS)
    )

    route_id = get_choice(
        "Route ID: ",
        ROUTE_IDS
    )

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    print(
        "\nVehicle types:"
    )

    print(
        ", ".join(VEHICLE_TYPES)
    )

    vehicle_type = get_choice(
        "Vehicle type: ",
        VEHICLE_TYPES
    )

    # --------------------------------------------------------
    # Price per seat
    # --------------------------------------------------------

    price_per_seat = (
        total_amount / seat_count
    )

    # --------------------------------------------------------
    # Distance category
    # --------------------------------------------------------

    distance_category = (
        calculate_distance_category(
            distance_km
        )
    )

    print(
        f"\nDistance category: "
        f"{distance_category}"
    )

    # --------------------------------------------------------
    # Return raw feature dictionary
    # --------------------------------------------------------

    booking = {

        "seat_count":
            seat_count,

        "ticket_price":
            ticket_price,

        "total_amount":
            total_amount,

        "total_seats":
            total_seats,

        "distance_km":
            distance_km,

        "base_fare":
            base_fare,

        "lead_time_days":
            lead_time_days,

        "booking_month":
            booking_month,

        "travel_month":
            travel_month,

        "booking_day_of_week":
            booking_day_of_week,

        "travel_day_of_week":
            travel_day_of_week,

        "booking_hour":
            booking_hour,

        "travel_hour":
            travel_hour,

        "price_per_seat":
            price_per_seat,

        "payment_method":
            payment_method,

        "route_id":
            route_id,

        "vehicle_type":
            vehicle_type,

        "distance_category":
            distance_category
    }

    return booking



# CREATE 56 FEATURES


def create_feature_vector(
    booking,
    model
):

    # --------------------------------------------------------
    # Start with numerical features
    # --------------------------------------------------------

    features = {

        "num__seat_count":
            booking["seat_count"],

        "num__ticket_price":
            booking["ticket_price"],

        "num__total_amount":
            booking["total_amount"],

        "num__total_seats":
            booking["total_seats"],

        "num__distance_km":
            booking["distance_km"],

        "num__base_fare":
            booking["base_fare"],

        "num__lead_time_days":
            booking["lead_time_days"],

        "num__booking_month":
            booking["booking_month"],

        "num__travel_month":
            booking["travel_month"],

        "num__booking_day_of_week":
            booking["booking_day_of_week"],

        "num__travel_day_of_week":
            booking["travel_day_of_week"],

        "num__booking_hour":
            booking["booking_hour"],

        "num__travel_hour":
            booking["travel_hour"],

        "num__price_per_seat":
            booking["price_per_seat"]
    }

    # --------------------------------------------------------
    # Payment method one-hot encoding
    # --------------------------------------------------------

    for payment in PAYMENT_METHODS:

        column = (
            "cat__payment_method_"
            + payment
        )

        features[column] = int(
            booking["payment_method"] == payment
        )

    # --------------------------------------------------------
    # Route one-hot encoding
    # --------------------------------------------------------

    for route in ROUTE_IDS:

        column = (
            "cat__route_id_"
            + route
        )

        features[column] = int(
            booking["route_id"] == route
        )

    # --------------------------------------------------------
    # Vehicle one-hot encoding
    # --------------------------------------------------------

    for vehicle in VEHICLE_TYPES:

        column = (
            "cat__vehicle_type_"
            + vehicle
        )

        features[column] = int(
            booking["vehicle_type"] == vehicle
        )

    # --------------------------------------------------------
    # Distance category one-hot encoding
    # --------------------------------------------------------

    for category in DISTANCE_CATEGORIES:

        column = (
            "cat__distance_category_"
            + category
        )

        features[column] = int(
            booking["distance_category"] == category
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [features]
    )

    # --------------------------------------------------------
    # VERY IMPORTANT:
    # Use exactly the feature order expected
    # by the trained Logistic Regression model.
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        expected_features = list(
            model.feature_names_in_
        )

        missing = [
            feature
            for feature in expected_features
            if feature not in X.columns
        ]

        if missing:

            raise ValueError(
                "\nMissing model features:\n"
                + "\n".join(missing)
            )

        X = X[
            expected_features
        ]

    return X



# PREDICT CANCELLATION


def predict_cancellation(
    model,
    X
):

    probability = (
        model
        .predict_proba(X)[0][1]
    )

    prediction = int(
        probability >= 0.50
    )

    return probability, prediction



# RISK LEVEL


def get_risk_level(
    probability
):

    if probability < 0.30:

        return "LOW"

    elif probability < 0.60:

        return "MEDIUM"

    else:

        return "HIGH"



# DISPLAY RESULT


def display_result(
    booking,
    probability,
    prediction
):

    risk_level = get_risk_level(
        probability
    )

    print("\n")
    
    print("CANCELLATION PREDICTION RESULT")
    

    print(
        f"\nRoute ID                : "
        f"{booking['route_id']}"
    )

    print(
        f"Vehicle Type            : "
        f"{booking['vehicle_type']}"
    )

    print(
        f"Seat Count              : "
        f"{booking['seat_count']}"
    )

    print(
        f"Payment Method          : "
        f"{booking['payment_method']}"
    )

    print(
        f"Lead Time               : "
        f"{booking['lead_time_days']} days"
    )

    print(
        f"Distance                : "
        f"{booking['distance_km']:.2f} km"
    )

    print(
        f"Total Amount            : "
        f"{booking['total_amount']:.2f}"
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"\nCancellation Probability : "
        f"{probability:.2%}"
    )

    print(
        f"Risk Level               : "
        f"{risk_level}"
    )

    if prediction == 1:

        print(
            "Prediction               : "
            "LIKELY TO CANCEL"
        )

    else:

        print(
            "Prediction               : "
            "LIKELY NOT TO CANCEL"
        )

    print(
        "\n"
        + "-" * 80
    )

    # --------------------------------------------------------
    # Business recommendation
    # --------------------------------------------------------

    if risk_level == "HIGH":

        print(
            "\nBusiness Recommendation:"
        )

        print(
            "Consider sending a booking "
            "confirmation/reminder to the customer."
        )

    elif risk_level == "MEDIUM":

        print(
            "\nBusiness Recommendation:"
        )

        print(
            "Monitor this booking and consider "
            "a reminder closer to travel date."
        )

    else:

        print(
            "\nBusiness Recommendation:"
        )

        print(
            "No immediate cancellation-risk "
            "intervention is required."
        )



# SAVE SINGLE PREDICTION


def save_prediction(
    booking,
    probability,
    prediction
):

    risk_level = get_risk_level(
        probability
    )

    output = pd.DataFrame(
        [
            {
                "route_id":
                    booking["route_id"],

                "vehicle_type":
                    booking["vehicle_type"],

                "seat_count":
                    booking["seat_count"],

                "ticket_price":
                    booking["ticket_price"],

                "total_amount":
                    booking["total_amount"],

                "payment_method":
                    booking["payment_method"],

                "distance_km":
                    booking["distance_km"],

                "lead_time_days":
                    booking["lead_time_days"],

                "cancellation_probability":
                    probability,

                "predicted_cancellation":
                    prediction,

                "risk_level":
                    risk_level
            }
        ]
    )

    output_path = os.path.join(
        REPORT_DIR,
        "new_booking_prediction.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPrediction saved:"
    )

    print(
        output_path
    )



# MAIN


def main():

    
    print("STEP 10.8 — NEW BOOKING CANCELLATION PREDICTION")
    

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Collect booking
    # --------------------------------------------------------

    booking = collect_booking_details()

    # --------------------------------------------------------
    # Create 56-feature vector
    # --------------------------------------------------------

    X = create_feature_vector(
        booking,
        model
    )

    print(
        f"\nPrediction feature count: "
        f"{X.shape[1]}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probability, prediction = (
        predict_cancellation(
            model,
            X
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_result(
        booking,
        probability,
        prediction
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_prediction(
        booking,
        probability,
        prediction
    )

    print("\n")
    
    print("STEP 10.8 COMPLETED")
    



# ENTRY POINT


if __name__ == "__main__":

    main()