import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Patient Flow Time Series Forecasting @ Rolling SARIMA",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# DARK BLUE THEME
# ============================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #071A33 !important;
}

[data-testid="stHeader"] {
    background-color: #071A33 !important;
}

[data-testid="stToolbar"] {
    background-color: #071A33 !important;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: #F8FFFC;
    border-radius: 24px;
    padding: 28px 32px;
    margin-bottom: 22px;
    border: 1px solid #BBD8E8;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
}

.hero h1 {
    color: #123C5A !important;
    margin: 0;
    font-size: 2.25rem;
}

.hero p {
    color: #58736C !important;
    margin: 8px 0 0;
    font-size: 1.05rem;
}

.card {
    background: #F8FFFC;
    border-radius: 20px;
    padding: 22px 24px;
    margin: 16px 0;
    border: 1px solid #BBD8E8;
    box-shadow: 0 6px 22px rgba(0,0,0,0.22);
}

.card-title {
    color: #123C5A !important;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.card-subtitle {
    color: #607870 !important;
    font-size: 0.96rem;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background: #F8FFFC !important;
    border: 1px solid #BBD8E8 !important;
    border-radius: 16px !important;
    padding: 12px 16px !important;
    box-shadow: 0 5px 18px rgba(0,0,0,0.20);
}

div[data-testid="stMetricLabel"] {
    color: #55736A !important;
}

div[data-testid="stMetricValue"] {
    color: #123C5A !important;
}

[data-testid="stFileUploader"] {
    background: #102C4D !important;
    border: 1px solid #396080 !important;
    border-radius: 16px !important;
    padding: 10px !important;
}

[data-testid="stFileUploader"] section {
    background: #102C4D !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #102C4D !important;
}

.stMarkdown, .stText, label {
    color: #EAF4FA;
}

.stDownloadButton button {
    border-radius: 10px;
}

.success-card {
    background: #E8F8F1;
    border-left: 5px solid #3C8C6A;
    color: #285B49;
    padding: 13px 16px;
    border-radius: 12px;
    margin: 12px 0;
}

.footer {
    color: #BBD0E1 !important;
    text-align: center;
    margin-top: 30px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🏥 Patient Flow Time Series Forecasting @ Rolling SARIMA</h1>
    <p>
        AI-assisted patient-flow monitoring, rolling model evaluation,
        and future patient-volume forecasting.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SETTINGS
# ============================================================
DATE_COL = "Date"
TARGET_COL = "Patient_Volume"

# FINALIZED MODEL
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)


# ============================================================
# DATA VALIDATION
# ============================================================
def load_and_validate(file, minimum_observations):

    try:
        df = pd.read_csv(file)

    except Exception as e:
        return None, [f"Could not read CSV: {e}"], []

    problems = []
    warnings_list = []

    missing_columns = [
        col for col in [DATE_COL, TARGET_COL]
        if col not in df.columns
    ]

    if missing_columns:

        problems.append(
            "Missing required column(s): "
            + ", ".join(missing_columns)
        )

        return None, problems, warnings_list

    df = df[[DATE_COL, TARGET_COL]].copy()

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce"
    )

    bad_dates = int(df[DATE_COL].isna().sum())
    bad_values = int(df[TARGET_COL].isna().sum())

    if bad_dates:
        problems.append(
            f"{bad_dates} invalid/missing date value(s)."
        )

    if bad_values:
        problems.append(
            f"{bad_values} invalid/missing "
            f"{TARGET_COL} value(s)."
        )

    if problems:
        return None, problems, warnings_list

    df = (
        df
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    duplicates = int(
        df[DATE_COL].duplicated().sum()
    )

    if duplicates:
        problems.append(
            f"{duplicates} duplicate date(s) found."
        )

    negatives = int(
        (df[TARGET_COL] < 0).sum()
    )

    if negatives:
        problems.append(
            f"{negatives} negative patient-volume "
            f"value(s) found."
        )

    if len(df) < minimum_observations:

        problems.append(
            f"Only {len(df)} observations found. "
            f"At least {minimum_observations} are required."
        )

    expected = pd.date_range(
        df[DATE_COL].min(),
        df[DATE_COL].max(),
        freq="D"
    )

    missing_days = expected.difference(
        df[DATE_COL]
    )

    if len(missing_days):

        warnings_list.append(
            f"{len(missing_days)} calendar date(s) "
            f"are missing."
        )

    if df[TARGET_COL].nunique() <= 1:

        problems.append(
            "Patient volume has no meaningful variation."
        )

    if problems:

        return None, problems, warnings_list

    return df, problems, warnings_list


# ============================================================
# MAKE TIME SERIES
# ============================================================
def make_series(df):

    return (
        df
        .set_index(DATE_COL)[TARGET_COL]
        .astype(float)
        .sort_index()
    )


# ============================================================
# MAPE
# ============================================================
def safe_mape(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


# ============================================================
# METRICS
# ============================================================
def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "MAPE": safe_mape(
            actual,
            predicted
        )
    }


# ============================================================
# FIT SARIMA
# ============================================================
def fit_model(series):

    model = SARIMAX(
        series,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    return model.fit(
        disp=False
    )


# ============================================================
# ROLLING SARIMA BACKTEST
# ============================================================
def rolling_backtest(
    series,
    test_days=None
):

    """
    One-step rolling SARIMA:

    predict
    ↓
    observe actual
    ↓
    update model
    ↓
    predict again
    """

    if test_days is None:

        test_days = max(
            7,
            int(len(series) * 0.20)
        )

    if len(series) <= test_days + 14:

        raise ValueError(
            "Not enough observations "
            "for rolling evaluation."
        )

    train = series.iloc[:-test_days]

    test = series.iloc[-test_days:]

    fitted = fit_model(train)

    predictions = []

    for date, actual_value in test.items():

        prediction = fitted.forecast(
            steps=1
        )

        predictions.append(
            float(prediction.iloc[0])
        )

        # UPDATE MODEL WITH ACTUAL VALUE
        fitted = fitted.extend(
            pd.Series(
                [actual_value],
                index=[date]
            )
        )

    predictions = pd.Series(
        predictions,
        index=test.index,
        name="Rolling SARIMA"
    )

    metrics = calculate_metrics(
        test,
        predictions
    )

    comparison = pd.DataFrame({

        "Actual": test,

        "Rolling SARIMA": predictions

    })

    return comparison, metrics


# ============================================================
# FUTURE FORECAST
# ============================================================
def future_forecast(
    series,
    horizon
):

    fitted = fit_model(series)

    forecast = fitted.forecast(
        steps=horizon
    )

    future_dates = pd.date_range(

        start=(
            series.index[-1]
            + pd.Timedelta(days=1)
        ),

        periods=horizon,

        freq="D"
    )

    values = np.maximum(
        np.asarray(
            forecast,
            dtype=float
        ),
        0
    )

    return pd.Series(
        values,
        index=future_dates,
        name="Forecast"
    )


# ============================================================
# TRAINING PLOT
# ============================================================
def show_training_plot(series):

    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    ax.plot(
        series.index,
        series.values,
        linewidth=1.8
    )

    ax.set_title(
        "Historical Patient Flow"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Patient Volume"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# BACKTEST PLOT
# ============================================================
def show_backtest_plot(
    comparison,
    title
):

    fig, ax = plt.subplots(
        figsize=(12, 4.8)
    )

    ax.plot(
        comparison.index,
        comparison["Actual"],
        label="Actual",
        linewidth=2
    )

    ax.plot(
        comparison.index,
        comparison["Rolling SARIMA"],
        label="Rolling SARIMA",
        linewidth=1.8
    )

    ax.set_title(title)

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Patient Volume"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# FUTURE FORECAST PLOT
# ============================================================
def show_future_plot(
    series,
    forecast
):

    history = series.tail(
        min(90, len(series))
    )

    fig, ax = plt.subplots(
        figsize=(12, 4.8)
    )

    ax.plot(
        history.index,
        history.values,
        label="Recent patient flow",
        linewidth=2
    )

    ax.plot(
        forecast.index,
        forecast.values,
        label="Future forecast",
        linewidth=2
    )

    ax.axvline(
        series.index[-1],
        linestyle="--",
        linewidth=1.2,
        label="Forecast starts"
    )

    ax.set_title(
        f"Future Patient Flow — "
        f"Next {len(forecast)} Days"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Patient Volume"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# STEP 1 — TRAINING DATA
# ============================================================
st.markdown("""
<div class="card">

    <div class="card-title">
        1️⃣ Upload training dataset
    </div>

    <div class="card-subtitle">

        Upload the historical CSV used for your
        finalized Rolling SARIMA workflow.

        Required columns:
        <b>Date</b> and <b>Patient_Volume</b>.

    </div>

</div>
""", unsafe_allow_html=True)


training_file = st.file_uploader(

    "Training dataset",

    type=["csv"],

    key="training_upload",

    label_visibility="collapsed"
)


if training_file is None:

    st.info(
        "Upload your historical training CSV to begin."
    )

    st.markdown(
        '<div class="footer">'
        '🏥 Rolling SARIMA • Patient Flow Forecasting'
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


training_df, training_problems, training_warnings = (
    load_and_validate(
        training_file,
        minimum_observations=90
    )
)


if training_df is None:

    st.error(
        "Training dataset could not be accepted."
    )

    for item in training_problems:

        st.write(
            "• " + item
        )

    st.stop()


for item in training_warnings:

    st.warning(item)


training_series = make_series(
    training_df
)


a, b, c = st.columns(3)


a.metric(
    "Training observations",
    f"{len(training_series):,}"
)


b.metric(
    "Training start",
    training_series.index.min().strftime(
        "%d %b %Y"
    )
)


c.metric(
    "Training end",
    training_series.index.max().strftime(
        "%d %b %Y"
    )
)


show_training_plot(
    training_series
)


# ============================================================
# STEP 2 — FINAL MODEL PERFORMANCE
# ============================================================
st.markdown("""
<div class="card">

    <div class="card-title">
        2️⃣ Finalized model — Rolling SARIMA
    </div>

    <div class="card-subtitle">

        One-day-ahead rolling forecasting
        with weekly seasonality.

    </div>

</div>
""", unsafe_allow_html=True)


with st.spinner(
    "Evaluating the finalized Rolling SARIMA model..."
):

    try:

        train_comparison, train_metrics = (
            rolling_backtest(
                training_series
            )
        )

    except Exception as e:

        st.error(
            f"Rolling SARIMA evaluation failed: {e}"
        )

        st.stop()


m1, m2, m3 = st.columns(3)


m1.metric(
    "MAE",
    f"{train_metrics['MAE']:.2f}"
)


m2.metric(
    "RMSE",
    f"{train_metrics['RMSE']:.2f}"
)


m3.metric(
    "MAPE",
    f"{train_metrics['MAPE']:.2f}%"
)


st.markdown(
    f"""
    <div class="success-card">

        <b>Final model:</b>
        Rolling SARIMA

        &nbsp; | &nbsp;

        <b>Order:</b>
        {ORDER}

        &nbsp; | &nbsp;

        <b>Seasonality:</b>
        {SEASONAL_ORDER[3]} days

        &nbsp; | &nbsp;

        <b>Evaluation:</b>
        final 20% rolling backtest

    </div>
    """,
    unsafe_allow_html=True
)


# FINAL MODEL TABLE
model_table = pd.DataFrame({

    "Model": [
        "Rolling SARIMA"
    ],

    "MAE": [
        round(
            train_metrics["MAE"],
            2
        )
    ],

    "RMSE": [
        round(
            train_metrics["RMSE"],
            2
        )
    ],

    "MAPE (%)": [
        round(
            train_metrics["MAPE"],
            2
        )
    ]

})


st.dataframe(

    model_table,

    use_container_width=True,

    hide_index=True

)


show_backtest_plot(

    train_comparison,

    "Rolling SARIMA — Actual vs Forecast"

)


# ============================================================
# STEP 3 — NEW / DUMMY DATA
# ============================================================
st.markdown("""
<div class="card">

    <div class="card-title">
        3️⃣ Upload new / dummy patient data
    </div>

    <div class="card-subtitle">

        Upload the latest patient-flow data.
        The finalized model will evaluate this
        new dataset and then forecast future
        patient volume.

    </div>

</div>
""", unsafe_allow_html=True)


new_file = st.file_uploader(

    "New patient-flow dataset",

    type=["csv"],

    key="new_upload",

    label_visibility="collapsed"
)


horizon = st.selectbox(

    "Future forecast horizon",

    [7, 14, 30, 60],

    index=2,

    format_func=lambda x:
        f"Next {x} days"

)


if new_file is None:

    st.info(
        "Upload your new/dummy CSV to continue."
    )


else:

    # ========================================================
    # NEW DATA VALIDATION
    # ========================================================

    new_df, new_problems, new_warnings = (
        load_and_validate(

            new_file,

            minimum_observations=21

        )
    )


    if new_df is None:

        st.error(
            "New dataset could not be accepted."
        )

        for item in new_problems:

            st.write(
                "• " + item
            )

        st.stop()


    for item in new_warnings:

        st.warning(item)


    new_series = make_series(
        new_df
    )


    n1, n2, n3 = st.columns(3)


    n1.metric(
        "New observations",
        f"{len(new_series):,}"
    )


    n2.metric(
        "Latest patient volume",
        f"{new_series.iloc[-1]:.0f}"
    )


    n3.metric(
        "Forecast horizon",
        f"{horizon} days"
    )


    # ========================================================
    # NEW DATA MAE / RMSE / MAPE
    # ========================================================

    with st.spinner(
        "Calculating new-data MAE, RMSE and MAPE..."
    ):

        try:

            # 30-day dummy data:
            # use final 7 days for evaluation

            if len(new_series) <= 45:

                evaluation_days = 7

            else:

                evaluation_days = max(
                    7,
                    int(
                        len(new_series)
                        * 0.20
                    )
                )


            new_comparison, new_metrics = (
                rolling_backtest(

                    new_series,

                    test_days=evaluation_days

                )
            )


        except Exception as e:

            new_comparison = None

            new_metrics = None

            st.error(

                "New-data performance could "
                "not be calculated. "
                f"Reason: {e}"

            )


    # ========================================================
    # DISPLAY NEW DATA METRICS
    # ========================================================

    if new_metrics is not None:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🎯 New dataset performance
            </div>

            <div class="card-subtitle">

                Rolling SARIMA performance on
                the held-out latest portion of
                the newly uploaded dataset.

            </div>

        </div>
        """, unsafe_allow_html=True)


        nm1, nm2, nm3 = st.columns(3)


        # NEW MAE
        nm1.metric(

            "NEW DATA — MAE",

            f"{new_metrics['MAE']:.2f}"

        )


        # NEW RMSE
        nm2.metric(

            "NEW DATA — RMSE",

            f"{new_metrics['RMSE']:.2f}"

        )


        # NEW MAPE
        nm3.metric(

            "NEW DATA — MAPE",

            f"{new_metrics['MAPE']:.2f}%"

        )


        # NEW MODEL TABLE
        new_model_table = pd.DataFrame({

            "Model": [
                "Rolling SARIMA"
            ],

            "MAE": [
                round(
                    new_metrics["MAE"],
                    2
                )
            ],

            "RMSE": [
                round(
                    new_metrics["RMSE"],
                    2
                )
            ],

            "MAPE (%)": [
                round(
                    new_metrics["MAPE"],
                    2
                )
            ]

        })


        st.dataframe(

            new_model_table,

            use_container_width=True,

            hide_index=True

        )


        # NEW DATA GRAPH
        show_backtest_plot(

            new_comparison,

            "New Data — Rolling SARIMA Performance"

        )


    # ========================================================
    # FUTURE FORECAST
    # ========================================================

    with st.spinner(
        "Generating future patient-flow forecast..."
    ):

        try:

            future = future_forecast(

                new_series,

                horizon

            )

        except Exception as e:

            st.error(
                f"Future forecast failed: {e}"
            )

            st.stop()


    st.markdown("""
    <div class="success-card">

        <b>🔮 Future forecast ready.</b>

        Rolling SARIMA has been fitted on the
        complete new/latest dataset and used
        to forecast future patient volume.

    </div>
    """, unsafe_allow_html=True)


    # FUTURE FORECAST GRAPH
    show_future_plot(

        new_series,

        future

    )


    # ========================================================
    # FUTURE FORECAST VALUES
    # ========================================================

    forecast_table = pd.DataFrame({

        "Date":
            future.index.strftime(
                "%Y-%m-%d"
            ),

        "Forecast Patient Volume":
            np.round(
                future.values
            ).astype(int)

    })


    with st.expander(
        "View future forecast values"
    ):

        st.dataframe(

            forecast_table,

            use_container_width=True,

            hide_index=True

        )


    # DOWNLOAD FORECAST
    st.download_button(

        "⬇️ Download future forecast CSV",

        data=
            forecast_table
            .to_csv(index=False)
            .encode("utf-8"),

        file_name=
            "rolling_sarima_patient_flow_forecast.csv",

        mime="text/csv"

    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(

    '<div class="footer">'
    '🏥 Patient Flow Time Series Forecasting • Rolling SARIMA'
    '</div>',

    unsafe_allow_html=True

)
