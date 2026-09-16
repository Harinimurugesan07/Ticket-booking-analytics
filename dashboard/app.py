
# =============================================================================
# TICKET BOOKING ANALYTICS
# STEP 12 — MODERN LIGHT BUSINESS INTELLIGENCE DASHBOARD
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TicketNow Analytics",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PROCESSED = BASE_DIR / "data" / "processed"
ANALYTICS_DIR = BASE_DIR / "reports" / "analytics"
ML_DIR = BASE_DIR / "reports" / "ml"
RECOMMENDATION_DIR = BASE_DIR / "reports" / "recommendations"


# =============================================================================
# LIGHT DASHBOARD THEME
# =============================================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f5f8fc;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.90);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e7edf5;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #102a43;
    }

    section[data-testid="stSidebar"] label {
        color: #52606d !important;
        font-weight: 500;
    }

    /* ---------- TOP BRAND ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 5px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: linear-gradient(135deg, #0b5cab, #21b6d7);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 22px;
        box-shadow: 0 8px 20px rgba(20, 116, 180, 0.20);
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        color: #102a43;
        letter-spacing: -0.5px;
    }

    .brand-sub {
        color: #829ab1;
        font-size: 12px;
        margin-top: -3px;
    }

    /* ---------- PAGE HEADER ---------- */

    .page-header {
        background: rgba(255,255,255,0.88);
        border: 1px solid #e6edf5;
        border-radius: 22px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(16, 42, 67, 0.05);
    }

    .page-title {
        color: #102a43;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }

    .page-subtitle {
        color: #829ab1;
        font-size: 14px;
        margin-top: 5px;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 20px;
        background: #e7f8f4;
        color: #087f5b;
        font-size: 12px;
        font-weight: 700;
        margin-top: 12px;
    }

    /* ---------- SECTION TITLES ---------- */

    .section-title {
        color: #102a43;
        font-size: 22px;
        font-weight: 750;
        margin: 24px 0 4px 0;
    }

    .section-description {
        color: #829ab1;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 7px 24px rgba(16, 42, 67, 0.055);
    }

    .kpi-card.primary {
        background: linear-gradient(135deg, #123f72, #126b9e);
        border: none;
    }

    .kpi-label {
        color: #627d98;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .primary .kpi-label {
        color: #d9f2ff;
    }

    .kpi-value {
        color: #102a43;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .primary .kpi-value {
        color: #ffffff;
    }

    .kpi-note {
        color: #829ab1;
        font-size: 11px;
        margin-top: 7px;
    }

    .primary .kpi-note {
        color: #c8eaf7;
    }

    /* ---------- PANELS ---------- */

    .dashboard-panel {
        background: #ffffff;
        border: 1px solid #e7edf5;
        border-radius: 20px;
        padding: 8px;
        box-shadow: 0 7px 24px rgba(16, 42, 67, 0.045);
    }

    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #d9e5f0;
        background: white;
        color: #1f4e79;
        font-weight: 650;
    }

    .stButton > button:hover {
        border-color: #20a4c9;
        color: #087f9c;
    }

    /* ---------- SELECTBOX / MULTISELECT ---------- */

    div[data-baseweb="select"] > div {
        border-radius: 11px;
        border-color: #dce6f0;
        background: white;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #9fb3c8;
        font-size: 12px;
        padding: 22px 0 5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

def load_csv(path):
    """Safely load a CSV file."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
    return pd.DataFrame()


def numeric_series(df, column):
    """Return a numeric Series, or an empty Series if unavailable."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def format_currency(value):
    if value is None or pd.isna(value):
        return "₹0"
    return f"₹{value:,.0f}"


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def section_header(title, description=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True,
        )


def kpi_card(label, value, note="", primary=False):
    cls = "kpi-card primary" if primary else "kpi-card"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_bookings(df):
    df = df.copy()

    for column in ["booking_date", "travel_date"]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def plot_config(fig, height=340):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color="#243b53",
        ),
        title=dict(
            font=dict(
                size=17,
                color="#102a43",
            )
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.75)",
            borderwidth=0,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#edf2f7",
            zeroline=False,
        ),
    )
    return fig


def show_plot(fig, height=340):
    fig = plot_config(fig, height)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def bool_series(series):
    """Convert common boolean-like CSV values safely."""
    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(
        ["true", "1", "yes", "y", "repeat", "high"]
    )


def normalize_probability(series):
    """Keep probabilities in percentage form."""
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values

    # If model exported 0-1 probabilities, convert to 0-100.
    if values.dropna().max() <= 1:
        return values * 100

    return values


# =============================================================================
# LOAD CORE DATA
# =============================================================================

bookings = prepare_bookings(
    load_csv(DATA_PROCESSED / "bookings_clean.csv")
)

customers = load_csv(
    DATA_PROCESSED / "customers_clean.csv"
)

trips = load_csv(
    DATA_PROCESSED / "trips_clean.csv"
)

routes = load_csv(
    DATA_PROCESSED / "routes_clean.csv"
)


# =============================================================================
# LOAD ANALYTICS OUTPUTS
# =============================================================================

customer_analytics = load_csv(
    ANALYTICS_DIR / "customer_analytics.csv"
)

rfm_segments = load_csv(
    ANALYTICS_DIR / "rfm_customer_segments.csv"
)

rfm_summary = load_csv(
    ANALYTICS_DIR / "rfm_segment_summary.csv"
)

customer_clv = load_csv(
    ANALYTICS_DIR / "customer_clv.csv"
)

route_performance = load_csv(
    ANALYTICS_DIR / "route_performance.csv"
)

route_occupancy = load_csv(
    ANALYTICS_DIR / "route_occupancy.csv"
)

route_revenue = load_csv(
    ANALYTICS_DIR / "route_revenue.csv"
)

vehicle_revenue = load_csv(
    ANALYTICS_DIR / "vehicle_revenue.csv"
)

vehicle_occupancy = load_csv(
    ANALYTICS_DIR / "vehicle_occupancy.csv"
)

monthly_booking_demand = load_csv(
    ANALYTICS_DIR / "monthly_booking_demand.csv"
)

monthly_revenue = load_csv(
    ANALYTICS_DIR / "monthly_revenue.csv"
)

monthly_cancellation = load_csv(
    ANALYTICS_DIR / "monthly_cancellation.csv"
)

cohort_retention = load_csv(
    ANALYTICS_DIR / "cohort_retention.csv"
)

customer_cancellation = load_csv(
    ANALYTICS_DIR / "customer_cancellation.csv"
)


# =============================================================================
# LOAD ML OUTPUTS
# =============================================================================

cancellation_risk = load_csv(
    ML_DIR / "cancellation_risk_scores.csv"
)

demand_forecast = load_csv(
    ML_DIR / "monthly_demand_forecast.csv"
)

demand_test_evaluation = load_csv(
    ML_DIR / "demand_test_evaluation.csv"
)

future_demand_analysis = load_csv(
    ML_DIR / "future_demand_analysis.csv"
)


# =============================================================================
# LOAD RECOMMENDATIONS
# =============================================================================

recommendations = load_csv(
    RECOMMENDATION_DIR / "business_recommendations.csv"
)

recommendation_summary = load_csv(
    RECOMMENDATION_DIR / "recommendation_summary.csv"
)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-icon">🎫</div>
        <div>
            <div class="brand-name">TicketNow</div>
            <div class="brand-sub">Business Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

pages = [
    "Executive Overview",
    "Booking Analytics",
    "Revenue Analytics",
    "Customer Analytics",
    "RFM Segmentation",
    "CLV Analytics",
    "Route Analytics",
    "Vehicle & Occupancy",
    "Cancellation Analytics",
    "ML Cancellation Risk",
    "Demand Forecast",
    "Business Recommendations",
]

page = st.sidebar.radio(
    "Dashboard",
    pages,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")


# =============================================================================
# FILTERS
# =============================================================================

selected_vehicle = []
selected_payment = []
selected_status = []
selected_route = []

if not bookings.empty:

    if "vehicle_type" in bookings.columns:
        vehicle_options = sorted(
            bookings["vehicle_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_vehicle = st.sidebar.multiselect(
            "Vehicle Type",
            vehicle_options,
            default=vehicle_options,
        )

    if "payment_method" in bookings.columns:
        payment_options = sorted(
            bookings["payment_method"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_payment = st.sidebar.multiselect(
            "Payment Method",
            payment_options,
            default=payment_options,
        )

    if "booking_status" in bookings.columns:
        status_options = sorted(
            bookings["booking_status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_status = st.sidebar.multiselect(
            "Booking Status",
            status_options,
            default=status_options,
        )

    if "trip_id" in bookings.columns and not trips.empty:
        if "route_id" in trips.columns:
            route_options = sorted(
                trips["route_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_route = st.sidebar.multiselect(
                "Route ID",
                route_options,
                default=route_options,
            )


# =============================================================================
# FILTER BOOKINGS
# =============================================================================

filtered_bookings = bookings.copy()

if not filtered_bookings.empty:

    if selected_vehicle and "vehicle_type" in filtered_bookings.columns:
        filtered_bookings = filtered_bookings[
            filtered_bookings["vehicle_type"]
            .astype(str)
            .isin(selected_vehicle)
        ]

    if selected_payment and "payment_method" in filtered_bookings.columns:
        filtered_bookings = filtered_bookings[
            filtered_bookings["payment_method"]
            .astype(str)
            .isin(selected_payment)
        ]

    if selected_status and "booking_status" in filtered_bookings.columns:
        filtered_bookings = filtered_bookings[
            filtered_bookings["booking_status"]
            .astype(str)
            .isin(selected_status)
        ]

    # Filter by route through trips when possible.
    if selected_route and "trip_id" in filtered_bookings.columns:
        if "route_id" in trips.columns:
            trip_route_map = trips[
                ["id", "route_id"]
            ].copy() if "id" in trips.columns else pd.DataFrame()

            if not trip_route_map.empty:
                trip_route_map["id"] = trip_route_map["id"].astype(str)
                trip_route_map["route_id"] = trip_route_map["route_id"].astype(str)

                temp = filtered_bookings.copy()
                temp["trip_id_str"] = temp["trip_id"].astype(str)

                valid_trip_ids = trip_route_map[
                    trip_route_map["route_id"].isin(selected_route)
                ]["id"].tolist()

                filtered_bookings = temp[
                    temp["trip_id_str"].isin(valid_trip_ids)
                ].drop(columns=["trip_id_str"])


# =============================================================================
# TOP HEADER
# =============================================================================

st.markdown(
    """
    <div class="page-header">
        <div class="page-title">Ticket Booking Analytics</div>
        <div class="page-subtitle">
            Business Intelligence, Customer Analytics & Machine Learning
        </div>
        <div class="status-pill">● Analytics system active</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# EXECUTIVE OVERVIEW
# =============================================================================

if page == "Executive Overview":

    section_header(
        "Executive Overview",
        "A visual summary of booking performance, revenue, customers and demand.",
    )

    if filtered_bookings.empty:

        st.warning("No booking data available for the selected filters.")

    else:

        total_bookings = len(filtered_bookings)

        status = (
            filtered_bookings["booking_status"]
            .astype(str)
            .str.upper()
            if "booking_status" in filtered_bookings.columns
            else pd.Series(dtype=str)
        )

        confirmed = int((status == "CONFIRMED").sum())
        cancelled = int((status == "CANCELLED").sum())

        revenue = (
            numeric_series(
                filtered_bookings,
                "total_amount",
            ).sum()
            if "total_amount" in filtered_bookings.columns
            else 0
        )

        customers_count = (
            filtered_bookings["customer_id"].nunique()
            if "customer_id" in filtered_bookings.columns
            else 0
        )

        cancellation_rate = safe_percentage(
            cancelled,
            total_bookings,
        )

        # KPI row
        cols = st.columns(5)

        with cols[0]:
            kpi_card(
                "Total Bookings",
                f"{total_bookings:,}",
                "All booking records",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "Confirmed",
                f"{confirmed:,}",
                "Successful bookings",
            )

        with cols[2]:
            kpi_card(
                "Total Revenue",
                format_currency(revenue),
                "Booking value",
            )

        with cols[3]:
            kpi_card(
                "Customers",
                f"{customers_count:,}",
                "Unique customers",
            )

        with cols[4]:
            kpi_card(
                "Cancellation Rate",
                f"{cancellation_rate:.2f}%",
                "Cancelled / total",
            )

        st.markdown("")

        # Main trend row
        left, right = st.columns([1.25, 1])

        with left:

            section_header(
                "Booking Trend",
                "Monthly booking volume across the selected data.",
            )

            if "booking_date" in filtered_bookings.columns:

                monthly = (
                    filtered_bookings.dropna(
                        subset=["booking_date"]
                    )
                    .assign(
                        booking_month=lambda x:
                        x["booking_date"]
                        .dt.to_period("M")
                        .astype(str)
                    )
                    .groupby("booking_month")
                    .size()
                    .reset_index(name="bookings")
                )

                fig = px.area(
                    monthly,
                    x="booking_month",
                    y="bookings",
                    markers=True,
                )

                show_plot(fig, 350)

        with right:

            section_header(
                "Booking Status",
                "Current distribution of booking outcomes.",
            )

            if "booking_status" in filtered_bookings.columns:

                status_df = (
                    filtered_bookings[
                        "booking_status"
                    ]
                    .astype(str)
                    .str.upper()
                    .value_counts()
                    .reset_index()
                )

                status_df.columns = [
                    "status",
                    "bookings",
                ]

                fig = px.pie(
                    status_df,
                    names="status",
                    values="bookings",
                    hole=0.58,
                )

                show_plot(fig, 350)

        # Revenue + vehicle row
        left, right = st.columns(2)

        with left:

            section_header(
                "Monthly Revenue",
                "Revenue generated by booking month.",
            )

            if (
                "booking_date" in filtered_bookings.columns
                and "total_amount" in filtered_bookings.columns
            ):

                monthly_rev = (
                    filtered_bookings.dropna(
                        subset=["booking_date"]
                    )
                    .assign(
                        booking_month=lambda x:
                        x["booking_date"]
                        .dt.to_period("M")
                        .astype(str),
                        amount=lambda x:
                        pd.to_numeric(
                            x["total_amount"],
                            errors="coerce",
                        ),
                    )
                    .groupby("booking_month")["amount"]
                    .sum()
                    .reset_index(name="revenue")
                )

                fig = px.bar(
                    monthly_rev,
                    x="booking_month",
                    y="revenue",
                )

                show_plot(fig, 330)

        with right:

            section_header(
                "Bookings by Vehicle",
                "Demand split across available vehicle types.",
            )

            if "vehicle_type" in filtered_bookings.columns:

                vehicle = (
                    filtered_bookings[
                        "vehicle_type"
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                vehicle.columns = [
                    "vehicle_type",
                    "bookings",
                ]

                fig = px.bar(
                    vehicle,
                    x="vehicle_type",
                    y="bookings",
                )

                show_plot(fig, 330)


# =============================================================================
# BOOKING ANALYTICS
# =============================================================================

elif page == "Booking Analytics":

    section_header(
        "Booking Analytics",
        "Analyze booking volume, status, ticket sales and payment behavior.",
    )

    if filtered_bookings.empty:
        st.warning("No booking data available.")
    else:

        status_series = (
            filtered_bookings["booking_status"]
            .astype(str)
            .str.upper()
            if "booking_status" in filtered_bookings.columns
            else pd.Series(dtype=str)
        )

        total = len(filtered_bookings)
        confirmed = int((status_series == "CONFIRMED").sum())
        cancelled = int((status_series == "CANCELLED").sum())
        pending = int((status_series == "PENDING").sum())

        if "seat_count" in filtered_bookings.columns:
            seats = numeric_series(
                filtered_bookings,
                "seat_count",
            ).sum()
        else:
            seats = 0

        cols = st.columns(5)

        for col, label, value in [
            (cols[0], "Bookings", f"{total:,}"),
            (cols[1], "Confirmed", f"{confirmed:,}"),
            (cols[2], "Cancelled", f"{cancelled:,}"),
            (cols[3], "Pending", f"{pending:,}"),
            (cols[4], "Tickets / Seats", f"{seats:,.0f}"),
        ]:
            with col:
                kpi_card(label, value)

        st.markdown("")

        left, right = st.columns(2)

        with left:

            section_header(
                "Daily Booking Volume",
                "Booking activity over time.",
            )

            if "booking_date" in filtered_bookings.columns:

                daily = (
                    filtered_bookings.dropna(
                        subset=["booking_date"]
                    )
                    .assign(
                        date=lambda x:
                        x["booking_date"].dt.date
                    )
                    .groupby("date")
                    .size()
                    .reset_index(name="bookings")
                )

                fig = px.line(
                    daily,
                    x="date",
                    y="bookings",
                    markers=True,
                )

                show_plot(fig, 350)

        with right:

            section_header(
                "Payment Method",
                "Number of bookings by payment method.",
            )

            if "payment_method" in filtered_bookings.columns:

                payment = (
                    filtered_bookings[
                        "payment_method"
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                payment.columns = [
                    "payment_method",
                    "bookings",
                ]

                fig = px.bar(
                    payment,
                    x="payment_method",
                    y="bookings",
                )

                show_plot(fig, 350)

        if "booking_status" in filtered_bookings.columns:

            section_header(
                "Booking Status by Vehicle",
                "Compare confirmed, cancelled and pending bookings.",
            )

            cross = pd.crosstab(
                filtered_bookings["vehicle_type"]
                if "vehicle_type" in filtered_bookings.columns
                else pd.Series(["All"] * len(filtered_bookings)),
                filtered_bookings["booking_status"].astype(str).str.upper(),
            ).reset_index()

            id_col = cross.columns[0]
            value_cols = [c for c in cross.columns if c != id_col]

            long_cross = cross.melt(
                id_vars=[id_col],
                value_vars=value_cols,
                var_name="status",
                value_name="bookings",
            )

            fig = px.bar(
                long_cross,
                x=id_col,
                y="bookings",
                color="status",
                barmode="group",
            )

            show_plot(fig, 370)


# =============================================================================
# REVENUE ANALYTICS
# =============================================================================

elif page == "Revenue Analytics":

    section_header(
        "Revenue Analytics",
        "Understand revenue generation, average booking value and payment contribution.",
    )

    if filtered_bookings.empty:
        st.warning("No booking data available.")
    else:

        amounts = numeric_series(
            filtered_bookings,
            "total_amount",
        )

        total_revenue = amounts.sum() if not amounts.empty else 0
        avg_booking = amounts.mean() if not amounts.empty else 0

        tickets = (
            numeric_series(
                filtered_bookings,
                "seat_count",
            ).sum()
            if "seat_count" in filtered_bookings.columns
            else 0
        )

        revenue_per_ticket = (
            total_revenue / tickets
            if tickets > 0
            else 0
        )

        cols = st.columns(3)

        with cols[0]:
            kpi_card(
                "Total Revenue",
                format_currency(total_revenue),
                "Selected booking data",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "Average Booking Value",
                format_currency(avg_booking),
                "Average transaction",
            )

        with cols[2]:
            kpi_card(
                "Revenue / Ticket",
                format_currency(revenue_per_ticket),
                "Revenue divided by seats",
            )

        st.markdown("")

        left, right = st.columns(2)

        with left:

            section_header(
                "Monthly Revenue",
                "Revenue trend across booking months.",
            )

            if not monthly_revenue.empty:

                fig = px.line(
                    monthly_revenue,
                    x="booking_month",
                    y="revenue",
                    markers=True,
                )

                show_plot(fig, 350)

            elif (
                "booking_date" in filtered_bookings.columns
                and "total_amount" in filtered_bookings.columns
            ):

                data = filtered_bookings.dropna(
                    subset=["booking_date"]
                ).copy()

                data["booking_month"] = (
                    data["booking_date"]
                    .dt.to_period("M")
                    .astype(str)
                )

                data["total_amount"] = pd.to_numeric(
                    data["total_amount"],
                    errors="coerce",
                )

                data = (
                    data.groupby("booking_month")["total_amount"]
                    .sum()
                    .reset_index(name="revenue")
                )

                fig = px.line(
                    data,
                    x="booking_month",
                    y="revenue",
                    markers=True,
                )

                show_plot(fig, 350)

        with right:

            section_header(
                "Revenue by Vehicle",
                "Revenue contribution by vehicle type.",
            )

            if not vehicle_revenue.empty:

                fig = px.bar(
                    vehicle_revenue,
                    x="vehicle_type",
                    y="revenue",
                )

                show_plot(fig, 350)

            elif (
                "vehicle_type" in filtered_bookings.columns
                and "total_amount" in filtered_bookings.columns
            ):

                data = (
                    filtered_bookings.assign(
                        amount=lambda x:
                        pd.to_numeric(
                            x["total_amount"],
                            errors="coerce",
                        )
                    )
                    .groupby("vehicle_type")["amount"]
                    .sum()
                    .reset_index(name="revenue")
                )

                fig = px.bar(
                    data,
                    x="vehicle_type",
                    y="revenue",
                )

                show_plot(fig, 350)

        if (
            "payment_method" in filtered_bookings.columns
            and "total_amount" in filtered_bookings.columns
        ):

            section_header(
                "Revenue by Payment Method",
                "Contribution of each payment method to total revenue.",
            )

            data = (
                filtered_bookings.assign(
                    amount=lambda x:
                    pd.to_numeric(
                        x["total_amount"],
                        errors="coerce",
                    )
                )
                .groupby("payment_method")["amount"]
                .sum()
                .reset_index(name="revenue")
            )

            fig = px.pie(
                data,
                names="payment_method",
                values="revenue",
                hole=0.5,
            )

            show_plot(fig, 360)


# =============================================================================
# CUSTOMER ANALYTICS
# =============================================================================

elif page == "Customer Analytics":

    section_header(
        "Customer Analytics",
        "Understand customer booking behavior, spending and repeat activity.",
    )

    if customer_analytics.empty:

        st.warning("customer_analytics.csv was not found.")

    else:

        total_customers = len(customer_analytics)

        if "is_repeat_customer" in customer_analytics.columns:
            repeat_customers = int(
                bool_series(
                    customer_analytics["is_repeat_customer"]
                ).sum()
            )
        else:
            repeat_customers = 0

        total_spend = (
            numeric_series(
                customer_analytics,
                "total_spend",
            ).sum()
            if "total_spend" in customer_analytics.columns
            else 0
        )

        avg_spend = (
            total_spend / total_customers
            if total_customers
            else 0
        )

        cols = st.columns(3)

        with cols[0]:
            kpi_card(
                "Total Customers",
                f"{total_customers:,}",
                "Customer base",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "Repeat Customers",
                f"{repeat_customers:,}",
                "Customers with repeat activity",
            )

        with cols[2]:
            kpi_card(
                "Average Customer Spend",
                format_currency(avg_spend),
                "Average lifetime spend",
            )

        st.markdown("")

        left, right = st.columns(2)

        with left:

            section_header(
                "Customer Booking Frequency",
                "Distribution of customers by number of bookings.",
            )

            if "total_bookings" in customer_analytics.columns:

                distribution = (
                    pd.to_numeric(
                        customer_analytics["total_bookings"],
                        errors="coerce",
                    )
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )

                distribution.columns = [
                    "booking_count",
                    "customers",
                ]

                fig = px.bar(
                    distribution,
                    x="booking_count",
                    y="customers",
                )

                show_plot(fig, 350)

        with right:

            section_header(
                "Repeat vs New Customers",
                "Customer retention view.",
            )

            repeat_data = pd.DataFrame(
                {
                    "segment": [
                        "Repeat Customers",
                        "Other Customers",
                    ],
                    "customers": [
                        repeat_customers,
                        max(total_customers - repeat_customers, 0),
                    ],
                }
            )

            fig = px.pie(
                repeat_data,
                names="segment",
                values="customers",
                hole=0.55,
            )

            show_plot(fig, 350)

        section_header(
            "Top Customers",
            "Highest-value customers based on total spend.",
        )

        if "total_spend" in customer_analytics.columns:

            top_customers = (
                customer_analytics
                .sort_values(
                    "total_spend",
                    ascending=False,
                )
                .head(20)
            )

            st.dataframe(
                top_customers,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# RFM SEGMENTATION
# =============================================================================

elif page == "RFM Segmentation":

    section_header(
        "RFM Customer Segmentation",
        "Segment customers using Recency, Frequency and Monetary behavior.",
    )

    if rfm_summary.empty:

        st.warning("RFM summary data not found.")

    else:

        left, right = st.columns(2)

        with left:

            if (
                "segment" in rfm_summary.columns
                and "total_revenue" in rfm_summary.columns
            ):

                section_header(
                    "Revenue by Segment",
                    "Revenue contribution from each customer segment.",
                )

                fig = px.pie(
                    rfm_summary,
                    names="segment",
                    values="total_revenue",
                    hole=0.48,
                )

                show_plot(fig, 350)

        with right:

            if (
                "segment" in rfm_summary.columns
                and "customers" in rfm_summary.columns
            ):

                section_header(
                    "Customers by Segment",
                    "Size of each RFM customer group.",
                )

                fig = px.bar(
                    rfm_summary,
                    x="segment",
                    y="customers",
                )

                show_plot(fig, 350)

        section_header(
            "RFM Segment Summary",
            "Aggregated performance of every customer segment.",
        )

        st.dataframe(
            rfm_summary,
            use_container_width=True,
            hide_index=True,
        )

        if not rfm_segments.empty:

            section_header(
                "Customer-Level RFM Segmentation",
                "Detailed RFM scores and segment assignment.",
            )

            st.dataframe(
                rfm_segments.head(200),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# CLV ANALYTICS
# =============================================================================

elif page == "CLV Analytics":

    section_header(
        "Customer Lifetime Value",
        "Measure historical customer value and projected 12-month value.",
    )

    if customer_clv.empty:

        st.warning("customer_clv.csv was not found.")

    else:

        historical = numeric_series(
            customer_clv,
            "historical_clv",
        )

        projected = numeric_series(
            customer_clv,
            "projected_12_month_clv",
        )

        cols = st.columns(3)

        with cols[0]:
            kpi_card(
                "Average Historical CLV",
                format_currency(historical.mean()),
                "Historical customer value",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "Average Projected 12M CLV",
                format_currency(projected.mean()),
                "Expected next 12 months",
            )

        with cols[2]:
            kpi_card(
                "Total Projected CLV",
                format_currency(projected.sum()),
                "Combined customer value",
            )

        st.markdown("")

        if "customer_value_segment" in customer_clv.columns:

            segment = (
                customer_clv.assign(
                    projected_clv=pd.to_numeric(
                        customer_clv[
                            "projected_12_month_clv"
                        ],
                        errors="coerce",
                    )
                )
                .groupby("customer_value_segment")
                .agg(
                    customers=(
                        "customer_id",
                        "count",
                    ) if "customer_id" in customer_clv.columns
                    else (
                        "projected_clv",
                        "count",
                    ),
                    projected_clv=(
                        "projected_clv",
                        "sum",
                    ),
                )
                .reset_index()
            )

            section_header(
                "Projected CLV by Segment",
                "Expected value across customer value segments.",
            )

            fig = px.bar(
                segment,
                x="customer_value_segment",
                y="projected_clv",
            )

            show_plot(fig, 370)

        section_header(
            "Highest Value Customers",
            "Customers with the highest projected lifetime value.",
        )

        top_clv = (
            customer_clv
            .sort_values(
                "projected_12_month_clv",
                ascending=False,
            )
            .head(20)
        )

        st.dataframe(
            top_clv,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# ROUTE ANALYTICS
# =============================================================================

elif page == "Route Analytics":

    section_header(
        "Route Analytics",
        "Analyze route demand, revenue and occupancy performance.",
    )

    if route_performance.empty:

        st.warning("route_performance.csv was not found.")

    else:

        left, right = st.columns(2)

        with left:

            section_header(
                "Top Routes by Demand",
                "Routes with the highest booking volume.",
            )

            top_demand = (
                route_performance
                .sort_values(
                    "bookings",
                    ascending=False,
                )
                .head(10)
            )

            fig = px.bar(
                top_demand,
                x="route",
                y="bookings",
            )

            show_plot(fig, 370)

        with right:

            section_header(
                "Top Routes by Revenue",
                "Routes generating the highest revenue.",
            )

            top_revenue = (
                route_performance
                .sort_values(
                    "revenue",
                    ascending=False,
                )
                .head(10)
            )

            fig = px.bar(
                top_revenue,
                x="route",
                y="revenue",
            )

            show_plot(fig, 370)

        if not route_occupancy.empty:

            section_header(
                "Top Routes by Occupancy",
                "Routes with the strongest capacity utilization.",
            )

            occupancy = (
                route_occupancy
                .sort_values(
                    "occupancy_percentage",
                    ascending=False,
                )
                .head(10)
            )

            fig = px.bar(
                occupancy,
                x="route",
                y="occupancy_percentage",
            )

            show_plot(fig, 370)

        section_header(
            "Route Performance",
            "Detailed route-level business metrics.",
        )

        st.dataframe(
            route_performance,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# VEHICLE & OCCUPANCY
# =============================================================================

elif page == "Vehicle & Occupancy":

    section_header(
        "Vehicle & Occupancy",
        "Compare vehicle demand, revenue and capacity utilization.",
    )

    if vehicle_occupancy.empty:

        st.warning("Vehicle occupancy data not found.")

    else:

        left, right = st.columns(2)

        with left:

            section_header(
                "Vehicle Occupancy",
                "Average capacity utilization by vehicle type.",
            )

            fig = px.bar(
                vehicle_occupancy,
                x="vehicle_type",
                y="occupancy_percentage",
            )

            show_plot(fig, 370)

        with right:

            section_header(
                "Vehicle Revenue",
                "Revenue generated by vehicle type.",
            )

            if not vehicle_revenue.empty:

                fig = px.bar(
                    vehicle_revenue,
                    x="vehicle_type",
                    y="revenue",
                )

                show_plot(fig, 370)

        section_header(
            "Vehicle Occupancy Data",
            "Detailed capacity utilization metrics.",
        )

        st.dataframe(
            vehicle_occupancy,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# CANCELLATION ANALYTICS
# =============================================================================

elif page == "Cancellation Analytics":

    section_header(
        "Cancellation Analytics",
        "Monitor cancellation trends and identify frequent cancellers.",
    )

    if not monthly_cancellation.empty:

        rates = numeric_series(
            monthly_cancellation,
            "cancellation_rate",
        )

        latest_rate = (
            rates.dropna().iloc[-1]
            if not rates.dropna().empty
            else 0
        )

        average_rate = rates.mean()

        cols = st.columns(2)

        with cols[0]:
            kpi_card(
                "Latest Cancellation Rate",
                f"{latest_rate:.2f}%",
                "Most recent period",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "Average Cancellation Rate",
                f"{average_rate:.2f}%",
                "Average across periods",
            )

        st.markdown("")

        section_header(
            "Monthly Cancellation Trend",
            "Cancellation rate over time.",
        )

        fig = px.line(
            monthly_cancellation,
            x="booking_month",
            y="cancellation_rate",
            markers=True,
        )

        show_plot(fig, 380)

    if not customer_cancellation.empty:

        section_header(
            "Frequent Cancellers",
            "Customers with repeated cancellation behavior.",
        )

        if "frequent_canceller" in customer_cancellation.columns:

            frequent = customer_cancellation[
                bool_series(
                    customer_cancellation[
                        "frequent_canceller"
                    ]
                )
            ]

            st.dataframe(
                frequent.head(100),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# ML CANCELLATION RISK
# =============================================================================

elif page == "ML Cancellation Risk":

    section_header(
        "Machine Learning — Cancellation Risk",
        "Predicted cancellation probability for individual bookings.",
    )

    if cancellation_risk.empty:

        st.warning("cancellation_risk_scores.csv was not found.")

    else:

        risk_column = next(
            (
                col
                for col in [
                    "risk_level",
                    "Risk Level",
                    "risk",
                ]
                if col in cancellation_risk.columns
            ),
            None,
        )

        probability_column = next(
            (
                col
                for col in [
                    "cancellation_probability",
                    "probability",
                    "risk_probability",
                ]
                if col in cancellation_risk.columns
            ),
            None,
        )

        if probability_column:

            probabilities = normalize_probability(
                cancellation_risk[
                    probability_column
                ]
            )

            cols = st.columns(3)

            with cols[0]:
                kpi_card(
                    "Average Risk",
                    f"{probabilities.mean():.2f}%",
                    "Average predicted probability",
                    primary=True,
                )

            with cols[1]:
                kpi_card(
                    "Maximum Risk",
                    f"{probabilities.max():.2f}%",
                    "Highest predicted probability",
                )

            with cols[2]:
                kpi_card(
                    "Minimum Risk",
                    f"{probabilities.min():.2f}%",
                    "Lowest predicted probability",
                )

            st.markdown("")

        left, right = st.columns(2)

        with left:

            if risk_column:

                section_header(
                    "Risk Distribution",
                    "Bookings grouped by model risk level.",
                )

                risk_counts = (
                    cancellation_risk[
                        risk_column
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                risk_counts.columns = [
                    "risk_level",
                    "bookings",
                ]

                fig = px.pie(
                    risk_counts,
                    names="risk_level",
                    values="bookings",
                    hole=0.5,
                )

                show_plot(fig, 350)

        with right:

            if probability_column:

                section_header(
                    "Risk Probability Distribution",
                    "Distribution of predicted cancellation probability.",
                )

                probabilities_df = pd.DataFrame(
                    {
                        "probability": probabilities.dropna()
                    }
                )

                fig = px.histogram(
                    probabilities_df,
                    x="probability",
                    nbins=20,
                )

                show_plot(fig, 350)

        section_header(
            "Cancellation Risk Records",
            "Bookings ranked by their predicted cancellation risk.",
        )

        risk_display = cancellation_risk.copy()

        if probability_column:
            risk_display["_risk_percent"] = normalize_probability(
                risk_display[probability_column]
            )
            risk_display = risk_display.sort_values(
                "_risk_percent",
                ascending=False,
            ).drop(columns=["_risk_percent"])

        st.dataframe(
            risk_display.head(200),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# DEMAND FORECAST
# =============================================================================

elif page == "Demand Forecast":

    section_header(
        "Demand Forecasting",
        "Historical booking demand, test forecasts and future predictions.",
    )

    if demand_forecast.empty:

        st.warning("monthly_demand_forecast.csv was not found.")

    else:

        demand_forecast = demand_forecast.copy()

        if "booking_date" in demand_forecast.columns:
            demand_forecast["booking_date"] = pd.to_datetime(
                demand_forecast["booking_date"],
                errors="coerce",
            )

        if "data_type" in demand_forecast.columns:

            historical = demand_forecast[
                demand_forecast["data_type"] == "Historical"
            ]

            test_forecast = demand_forecast[
                demand_forecast["data_type"] == "Test Forecast"
            ]

            future = demand_forecast[
                demand_forecast["data_type"] == "Future Forecast"
            ]

        else:

            historical = demand_forecast
            test_forecast = pd.DataFrame()
            future = pd.DataFrame()

        cols = st.columns(3)

        with cols[0]:
            kpi_card(
                "Historical Months",
                f"{len(historical):,}",
                "Observed demand periods",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "Test Forecast Months",
                f"{len(test_forecast):,}",
                "Model evaluation periods",
            )

        with cols[2]:
            kpi_card(
                "Future Forecast Months",
                f"{len(future):,}",
                "Predicted future periods",
            )

        st.markdown("")

        section_header(
            "Historical Demand & Forecast",
            "Compare actual booking demand with model predictions.",
        )

        if (
            "booking_date" in demand_forecast.columns
            and "demand" in demand_forecast.columns
        ):

            fig = px.line(
                demand_forecast,
                x="booking_date",
                y="demand",
                color="data_type"
                if "data_type" in demand_forecast.columns
                else None,
                markers=True,
            )

            show_plot(fig, 420)

        if not future.empty:

            section_header(
                "Future Demand Forecast",
                "Expected booking demand for upcoming periods.",
            )

            st.dataframe(
                future,
                use_container_width=True,
                hide_index=True,
            )

        if not demand_test_evaluation.empty:

            section_header(
                "Forecast Test Evaluation",
                "Model accuracy and test-period evaluation metrics.",
            )

            st.dataframe(
                demand_test_evaluation,
                use_container_width=True,
                hide_index=True,
            )

        if not future_demand_analysis.empty:

            section_header(
                "Future Demand Business Analysis",
                "Business interpretation of predicted demand.",
            )

            st.dataframe(
                future_demand_analysis,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# BUSINESS RECOMMENDATIONS
# =============================================================================

elif page == "Business Recommendations":

    section_header(
        "Business Recommendation Engine",
        "Analytics-driven recommendations for operational and commercial decisions.",
    )

    if recommendations.empty:

        st.warning("business_recommendations.csv was not found.")

    else:

        total_recommendations = len(recommendations)

        if "priority" in recommendations.columns:

            high_priority = int(
                recommendations[
                    "priority"
                ]
                .astype(str)
                .str.upper()
                .eq("HIGH")
                .sum()
            )

        else:
            high_priority = 0

        cols = st.columns(2)

        with cols[0]:
            kpi_card(
                "Total Recommendations",
                f"{total_recommendations:,}",
                "Generated recommendations",
                primary=True,
            )

        with cols[1]:
            kpi_card(
                "High Priority",
                f"{high_priority:,}",
                "Requires immediate attention",
            )

        st.markdown("")

        left, right = st.columns(2)

        with left:

            if "recommendation_type" in recommendations.columns:

                section_header(
                    "Recommendations by Type",
                    "Distribution of recommendation categories.",
                )

                rec_summary = (
                    recommendations[
                        "recommendation_type"
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                rec_summary.columns = [
                    "recommendation_type",
                    "count",
                ]

                fig = px.bar(
                    rec_summary,
                    x="recommendation_type",
                    y="count",
                )

                show_plot(fig, 350)

        with right:

            if "priority" in recommendations.columns:

                section_header(
                    "Recommendation Priority",
                    "Priority distribution.",
                )

                priority_summary = (
                    recommendations[
                        "priority"
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                priority_summary.columns = [
                    "priority",
                    "count",
                ]

                fig = px.pie(
                    priority_summary,
                    names="priority",
                    values="count",
                    hole=0.5,
                )

                show_plot(fig, 350)

        section_header(
            "Recommendations",
            "Detailed business recommendations generated from analytics.",
        )

        st.dataframe(
            recommendations.head(200),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        TicketNow Analytics &nbsp;•&nbsp;
        Business Intelligence + Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
