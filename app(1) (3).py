import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG + STYLE
# ============================================================
st.set_page_config(
    page_title="Patient Flow Time Series Forecasting @ Rolling SARIMA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: #dff5ec;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: rgba(255,255,255,0.92);
        padding: 2rem 2.2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(31, 78, 64, 0.08);
        border: 1px solid rgba(70,130,100,0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        color: #174d3b;
    }

    .hero p {
        margin: .55rem 0 0 0;
        color: #55736a;
        font-size: 1.05rem;
    }

    .section-card {
        background: rgba(255,255,255,0.92);
        padding: 1.35rem 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 6px 24px rgba(31, 78, 64, 0.06);
    }

    .section-title {
        color: #174d3b;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: .25rem;
    }

    .section-subtitle {
        color: #6b827a;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 12px 16px;
        border: 1px solid rgba(70,130,100,0.12);
        box-shadow: 0 4px 15px rgba(31,78,64,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #55736a;
    }

    div[data-testid="stMetricValue"] {
        color: #174d3b;
    }

    .success-box {
        background: #eefaf5;
        border-left: 5px solid #3c8c6a;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: #285b49;
    }

    .warning-box {
        background: #fff8e8;
        border-left: 5px solid #d7a83d;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        color: #6d5720;
    }

    .footer {
        text-align: center;
        color: #6b827a;
        margin-top: 2rem;
        font-size: .9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🏥 Patient Flow Time Series Forecasting @ Rolling SARIMA</h1>
    <p>
        A practical forecasting dashboard for monitoring patient volume,
        evaluating Rolling SARIMA performance, and generating future forecasts.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
DATE_COL = "Date"
TARGET_COL = "Patient_Volume"

# Selected SARIMA structure from the finalized workflow.
# These can be changed here if a different order is selected during model tuning.
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 7)


def validate_forecasting_data(data, min_days=90):
    """Validate the two required columns and basic time-series quality."""
    problems = []
    warnings_list = []

    if DATE_COL not in data.columns:
        problems.append(f"Missing required column: {DATE_COL}")

    if TARGET_COL not in data.columns:
        problems.append(f"Missing required column: {TARGET_COL}")

    if problems:
        return False, None, problems, warnings_list

    check = data[[DATE_COL, TARGET_COL]].copy()

    check[DATE_COL] = pd.to_datetime(check[DATE_COL], errors="coerce")
    check[TARGET_COL] = pd.to_numeric(check[TARGET_COL], errors="coerce")

    invalid_dates = check[DATE_COL].isna().sum()
    missing_target = check[TARGET_COL].isna().sum()

    if invalid_dates:
        problems.append(f"{invalid_dates} invalid or missing dates found.")

    if missing_target:
        problems.append(
            f"{missing_target} missing/non-numeric {TARGET_COL} values found."
        )

    if check[DATE_COL].isna().any():
        return False, None, problems, warnings_list

    check = check.dropna(subset=[TARGET_COL]).sort_values(DATE_COL)

    duplicate_dates = check[DATE_COL].duplicated().sum()
    if duplicate_dates:
        problems.append(f"{duplicate_dates} duplicate dates found.")

    negative_values = (check[TARGET_COL] < 0).sum()
    if negative_values:
        problems.append(f"{negative_values} negative {TARGET_COL} values found.")

    if len(check) < min_days:
        problems.append(
            f"Only {len(check)} observations found. "
            f"At least {min_days} observations are recommended."
        )

    expected_dates = pd.date_range(
        check[DATE_COL].min(),
        check[DATE_COL].max(),
        freq="D"
    )
    missing_dates = expected_dates.difference(check[DATE_COL])

    if len(missing_dates):
        warnings_list.append(
            f"{len(missing_dates)} calendar date(s) are missing. "
            "The model will use the observations supplied."
        )

    if check[TARGET_COL].nunique() <= 1:
        problems.append(f"{TARGET_COL} has no meaningful variation.")

    return len(problems) == 0, check, problems, warnings_list


def make_series(data):
    """Prepare a clean daily patient-volume series."""
    ts = (
        data.set_index(DATE_COL)[TARGET_COL]
        .astype(float)
        .sort_index()
        .dropna()
    )
    return ts


def safe_mape(actual, forecast):
    """MAPE that ignores zero actual values."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs((actual[mask] - forecast[mask]) / actual[mask])
    ) * 100


def fit_sarima(ts, order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER):
    """Fit SARIMA quietly."""
    model = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def rolling_sarima_backtest(
    ts,
    test_size=0.20,
    order=DEFAULT_ORDER,
    seasonal_order=DEFAULT_SEASONAL_ORDER
):
    """
    One-step-ahead rolling SARIMA:
    forecast one day -> observe actual -> update model -> repeat.
    """
    split_point = int(len(ts) * (1 - test_size))

    train = ts.iloc[:split_point]
    test = ts.iloc[split_point:]

    if len(train) < 60 or len(test) < 10:
        raise ValueError("Not enough observations for a reliable rolling backtest.")

    fitted = fit_sarima(train, order, seasonal_order)

    predictions = []

    for i in range(len(test)):
        prediction = fitted.forecast(steps=1)
        predictions.append(float(prediction.iloc[0]))

        # Update with the newly observed actual value.
        fitted = fitted.extend(test.iloc[[i]])

    predictions = pd.Series(
        predictions,
        index=test.index,
        name="Rolling_SARIMA_Forecast"
    )

    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = safe_mape(test, predictions)

    results = pd.DataFrame({
        "Actual": test,
        "Rolling SARIMA": predictions
    })

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    return results, metrics, train, test


def future_rolling_sarima(
    ts,
    horizon=30,
    order=DEFAULT_ORDER,
    seasonal_order=DEFAULT_SEASONAL_ORDER
):
    """
    Fit the finalized SARIMA model on all available historical data
    and forecast the requested future horizon.
    """
    fitted = fit_sarima(ts, order, seasonal_order)
    forecast = fitted.forecast(steps=horizon)

    future_dates = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    return pd.Series(
        np.maximum(np.asarray(forecast, dtype=float), 0),
        index=future_dates,
        name="Forecast"
    )


def plot_training_data(ts):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(ts.index, ts.values, linewidth=1.7)
    ax.set_title("Historical Patient Flow", fontsize=15, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Patient Volume")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_backtest(results):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        results.index,
        results["Actual"],
        label="Actual",
        linewidth=2
    )
    ax.plot(
        results.index,
        results["Rolling SARIMA"],
        label="Rolling SARIMA",
        linewidth=1.8
    )
    ax.set_title("Rolling SARIMA: Actual vs Forecast", fontsize=15, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Patient Volume")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_future_forecast(ts, forecast):
    history = ts.tail(min(90, len(ts)))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        history.index,
        history.values,
        label="Recent historical flow",
        linewidth=2
    )
    ax.plot(
        forecast.index,
        forecast.values,
        label="Future Rolling SARIMA forecast",
        linewidth=2.2
    )
    ax.axvline(
        ts.index[-1],
        linestyle="--",
        linewidth=1.2,
        label="Forecast start"
    )
    ax.set_title(
        f"Patient Flow Forecast: Next {len(forecast)} Days",
        fontsize=15,
        pad=12
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Patient Volume")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# STEP 1 — TRAINING DATASET
# ============================================================
st.markdown("""
<div class="section-card">
<div class="section-title">1. Upload training dataset</div>
<div class="section-subtitle">
Upload the historical patient-flow CSV used to train and evaluate the model.
</div>
</div>
""", unsafe_allow_html=True)

training_file = st.file_uploader(
    "Training CSV",
    type=["csv"],
    key="training_file",
    label_visibility="collapsed"
)

if training_file is None:
    st.info(
        "👆 Upload your historical training CSV to start the forecasting workflow. "
        "Required columns: Date and Patient_Volume."
    )

    st.markdown("""
    <div class="section-card">
        <div class="section-title">What this app does</div>
        <p>
        <b>Train → Validate → Rolling SARIMA backtest → Evaluate → Forecast.</b>
        </p>
        <p style="color:#6b827a;">
        The dashboard keeps the technical workflow behind the scenes and
        presents only the final model, key performance metrics and useful forecasts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">Rolling SARIMA • Patient Flow Forecasting</div>',
        unsafe_allow_html=True
    )
    st.stop()

# Read training file.
try:
    raw_training = pd.read_csv(training_file)
except Exception as e:
    st.error(f"Could not read the training CSV: {e}")
    st.stop()

valid, training_data, problems, validation_warnings = validate_forecasting_data(
    raw_training
)

if not valid:
    st.error("Training dataset needs attention.")
    for problem in problems:
        st.write(f"• {problem}")
    st.stop()

for warning in validation_warnings:
    st.warning(warning)

training_ts = make_series(training_data)

# Compact dataset summary — no raw dataframe dump.
c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(training_ts):,}")
c2.metric("Start date", training_ts.index.min().strftime("%d %b %Y"))
c3.metric("End date", training_ts.index.max().strftime("%d %b %Y"))

st.markdown("""
<div class="section-card">
<div class="section-title">📈 Training data overview</div>
<div class="section-subtitle">
Historical patient volume used by the forecasting model.
</div>
</div>
""", unsafe_allow_html=True)

plot_training_data(training_ts)

# ============================================================
# STEP 2 — FINAL ROLLING SARIMA
# ============================================================
st.markdown("""
<div class="section-card">
<div class="section-title">2. Final model — Rolling SARIMA</div>
<div class="section-subtitle">
The model makes a one-day-ahead prediction, receives the actual observation,
updates itself, and then predicts the next day.
</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Training and evaluating Rolling SARIMA..."):
    try:
        backtest_results, metrics, train_ts, test_ts = rolling_sarima_backtest(
            training_ts,
            test_size=0.20,
            order=DEFAULT_ORDER,
            seasonal_order=DEFAULT_SEASONAL_ORDER
        )
    except Exception as e:
        st.error(f"Rolling SARIMA could not be fitted: {e}")
        st.stop()

# Main metrics — MAPE is deliberately prominent.
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics['MAE']:.2f}")
m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

st.markdown("""
<div class="success-box">
<b>Final model:</b> Rolling SARIMA
&nbsp; | &nbsp;
<b>Seasonality:</b> 7 days
&nbsp; | &nbsp;
<b>Evaluation:</b> 20% hold-out with rolling one-step forecasting
</div>
""", unsafe_allow_html=True)

# Only the finalized model table is shown.
final_model_table = pd.DataFrame({
    "Model": ["Rolling SARIMA"],
    "MAE": [round(metrics["MAE"], 2)],
    "RMSE": [round(metrics["RMSE"], 2)],
    "MAPE (%)": [round(metrics["MAPE"], 2)]
})

st.dataframe(
    final_model_table,
    use_container_width=True,
    hide_index=True
)

plot_backtest(backtest_results)

with st.expander("View forecast details"):
    detail = backtest_results.copy()
    detail.index.name = "Date"
    st.dataframe(
        detail.round(2),
        use_container_width=True
    )

# ============================================================
# STEP 3 — NEW DATA + FUTURE FORECAST
# ============================================================
st.markdown("""
<div class="section-card">
<div class="section-title">3. Forecast new patient flow</div>
<div class="section-subtitle">
Now upload a new/latest CSV. The finalized Rolling SARIMA model will use
the new history and generate the requested future forecast.
</div>
</div>
""", unsafe_allow_html=True)

new_file = st.file_uploader(
    "New / latest patient-flow CSV",
    type=["csv"],
    key="new_file",
    label_visibility="collapsed"
)

horizon = st.selectbox(
    "Forecast horizon",
    options=[7, 14, 30, 60],
    index=2,
    format_func=lambda x: f"Next {x} days"
)

if new_file is None:
    st.info(
        "📤 Upload a new or dummy patient-flow CSV with Date and Patient_Volume "
        "to generate the future forecast."
    )
else:
    try:
        raw_new = pd.read_csv(new_file)
    except Exception as e:
        st.error(f"Could not read the new CSV: {e}")
        st.stop()

    valid_new, new_data, new_problems, new_warnings = validate_forecasting_data(
        raw_new,
        min_days=30
    )

    if not valid_new:
        st.error("New dataset needs attention.")
        for problem in new_problems:
            st.write(f"• {problem}")
        st.stop()

    for warning in new_warnings:
        st.warning(warning)

    new_ts = make_series(new_data)

    n1, n2, n3 = st.columns(3)
    n1.metric("New observations", f"{len(new_ts):,}")
    n2.metric("Latest patient volume", f"{new_ts.iloc[-1]:.0f}")
    n3.metric("Forecast horizon", f"{horizon} days")

    with st.spinner("Generating future patient-flow forecast..."):
        try:
            future = future_rolling_sarima(
                new_ts,
                horizon=horizon,
                order=DEFAULT_ORDER,
                seasonal_order=DEFAULT_SEASONAL_ORDER
            )
        except Exception as e:
            st.error(f"Future forecast could not be generated: {e}")
            st.stop()

    st.markdown("""
    <div class="success-box">
    <b>Forecast ready.</b> The model has been fitted using the complete
    uploaded history and is forecasting future patient volume.
    </div>
    """, unsafe_allow_html=True)

    plot_future_forecast(new_ts, future)

    forecast_table = pd.DataFrame({
        "Date": future.index,
        "Forecast Patient Volume": np.round(future.values, 0).astype(int)
    })

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )

    # Download is useful, without cluttering the page.
    csv_output = forecast_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download forecast CSV",
        data=csv_output,
        file_name="rolling_sarima_patient_flow_forecast.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">🏥 Patient Flow Forecasting • Rolling SARIMA</div>',
    unsafe_allow_html=True
)
