An end-to-end Business Intelligence + Machine Learning project for
analyzing ticket-booking data, customer behavior, revenue, routes,
vehicle utilization, cancellations, demand, and business opportunities.

The project combines Python, Pandas, NumPy, Scikit-learn, Plotly, and
Streamlit to turn booking data into interactive analytics and
actionable business insights.

📌 Project Overview

The project covers:

Data generation and validation

Data cleaning and preparation

Exploratory and business analytics

Booking and revenue analytics

Customer behavior analysis

RFM customer segmentation

Customer Lifetime Value (CLV)

Route performance analysis

Vehicle and occupancy analysis

Cancellation analysis

Machine Learning cancellation-risk prediction

Monthly demand forecasting

Future demand analysis

Business recommendation generation

Interactive Streamlit BI dashboard

🎯 Objectives

Understand booking patterns and customer behavior.

Measure booking and revenue performance.

Identify repeat and high-value customers.

Segment customers using Recency, Frequency, and Monetary behavior.

Estimate Customer Lifetime Value.

Analyze route demand, revenue, and occupancy.

Compare vehicle demand and utilization.

Identify cancellation patterns.

Predict cancellation risk for bookings.

Forecast future monthly booking demand.

Convert analytics and ML results into business recommendations.

Present insights through an interactive dashboard.

🏗️ Project Workflow

Booking Data
    ↓
Data Validation
    ↓
Data Cleaning & Processing
    ↓
Exploratory / Business Analytics
    ↓
Customer Analytics ──→ RFM ──→ CLV
    ↓
Route / Revenue / Vehicle / Cancellation Analysis
    ↓
Machine Learning
    ├── Cancellation Risk Prediction
    └── Demand Forecasting
    ↓
Business Recommendation Engine
    ↓
Streamlit BI Dashboard

📁 Project Structure

ticket-booking-analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── bookings_clean.csv
│   │   ├── customers_clean.csv
│   │   ├── trips_clean.csv
│   │   └── routes_clean.csv
│   └── ml/
│
├── src/
│   ├── data/
│   ├── validation/
│   ├── analytics/
│   └── ml/
│
└── reports/
    ├── analytics/
    ├── ml/
    └── recommendations/

📊 Analytics Modules

1. Booking Analytics

Analyzes:

Total bookings

Confirmed bookings

Cancelled bookings

Pending bookings

Seats/tickets booked

Daily booking volume

Booking trends

Payment-method distribution

2. Revenue Analytics

Measures:

Total revenue

Average booking value

Revenue per ticket

Monthly revenue

Revenue by vehicle type

Revenue by payment method

3. Customer Analytics

Provides:

Total customers

Repeat customers

Average customer spending

Booking frequency

Top customers by spending

Customer-level booking behavior

4. RFM Segmentation

Customers are segmented using:

Recency --- how recently they booked

Frequency --- how often they booked

Monetary --- how much they spent

The dashboard displays segment revenue, customer counts, segment
summaries, and customer-level RFM records.

5. Customer Lifetime Value

CLV analysis includes:

Historical CLV

Projected 12-month CLV

Total projected CLV

Customer value segments

Highest-value customers

6. Route Analytics

Analyzes:

Booking demand by route

Revenue by route

Route performance

Route occupancy

Operational demand patterns

7. Vehicle & Occupancy Analytics

Compares vehicle categories using:

Booking demand

Revenue

Occupancy percentage

Capacity utilization

8. Cancellation Analytics

Includes:

Monthly cancellation rate

Average cancellation rate

Cancellation trends

Frequent cancellers

Customer-level cancellation behavior

🤖 Machine Learning

Cancellation Risk Prediction

The project uses Machine Learning to estimate the probability that a
booking may be cancelled.

Models include:

Logistic Regression

Random Forest

Features can include booking, trip, timing, payment, distance, route,
vehicle, and lead-time information.

The dashboard provides:

Risk distribution

Average predicted risk

Maximum predicted risk

Minimum predicted risk

Booking-level risk records

A separate scoring workflow can also be used for a new booking.

Demand Forecasting

Monthly booking demand is divided into:

Historical demand

Test forecast

Future forecast

The dashboard displays historical demand, test predictions, future
predictions, evaluation results, and future-demand business analysis.

Evaluation metrics include:

MAE

RMSE

R²

MAPE

The current forecasting experiment shows substantial prediction error,
so the forecasting model should be considered experimental and improved
before production use.

💡 Business Recommendation Engine

Analytics and ML outputs are converted into structured recommendations.

Recommendation records can contain:

Recommendation type

Entity type

Entity ID

Priority

Risk level

Probability

Recommendation

Business reason

This connects technical analytics with practical business decision
support.

🖥️ Streamlit Dashboard

The dashboard contains:

Executive Overview

Booking Analytics

Revenue Analytics

Customer Analytics

RFM Segmentation

CLV Analytics

Route Analytics

Vehicle & Occupancy

Cancellation Analytics

ML Cancellation Risk

Demand Forecast

Business Recommendations

Dashboard Features

Modern light BI-style interface

KPI cards

Interactive Plotly charts

Sidebar navigation

Vehicle filters

Payment-method filters

Booking-status filters

Route filters

Customer tables

ML prediction results

Forecast results

Business recommendations

🛠️ Technologies

Technology      Purpose

Python          Core development
Pandas          Data processing
NumPy           Numerical analysis
Scikit-learn    Machine Learning
Plotly          Interactive visualization
Streamlit       BI dashboard
CSV             Data/output storage
Joblib/Pickle   Model persistence

