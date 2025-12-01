import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import io

from src.generate_synthetic_data import generate_monthly_series
from src.data_prep import load_and_prepare
from src.train_test_split import time_series_split
from src.model_sarimax import fit_sarimax_auto, forecast_sarimax
from src.evaluate import evaluate
from src.narrative import generate_narrative

def format_large_number(num):
    """Format large numbers to be more readable."""
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.2f}"

# Page configuration
st.set_page_config(
    page_title="Financial Forecast Story",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Corporate styling
st.markdown("""
<style>
    .corporate-header {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .section-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
    }
    .primary-button {
        background: #0066cc;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        width: 100%;
    }
    .secondary-button {
        background: #6c757d;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .status-success {
        color: #28a745;
        font-weight: 500;
    }
    .status-info {
        color: #17a2b8;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Corporate header
st.markdown("""
<div class="corporate-header">
    <h1 style="color: #333; margin: 0; font-size: 2rem; font-weight: 600;">Financial Forecast Analysis</h1>
    <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1rem;">
        SARIMAX Time Series Forecasting Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize variables for file processing
selected_df = None
date_col = None
value_col = None
temp_df = None

# Data Input Section
st.subheader("Data Input")

uploaded_file = st.file_uploader(
    "Upload CSV file with time series data",
    type="csv",
    help="File should contain date and numeric value columns"
)

if not uploaded_file:
    st.markdown('<p style="color: #666; font-style: italic;">No file uploaded. System will use synthetic data for demonstration.</p>', unsafe_allow_html=True)

if uploaded_file:
    # Clear previous results on new file upload
    current_file_name = uploaded_file.name
    if 'last_file_name' not in st.session_state or st.session_state.last_file_name != current_file_name:
        st.session_state.results = None
        st.session_state.last_file_name = current_file_name

    # Progress indicator
    with st.spinner("Analyzing your data..."):
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

    # Read and parse file
    try:
        uploaded_file.seek(0)
        file_content = uploaded_file.read()
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8', errors='ignore')

        if not file_content.strip():
            st.error("Uploaded file is empty.")
            st.stop()

        # Try different CSV parsing options
        temp_df = None
        separators = [',', ';', '\t']

        for sep in separators:
            try:
                temp_df = pd.read_csv(io.StringIO(file_content), sep=sep)
                if not temp_df.empty and len(temp_df.columns) >= 2:
                    break
            except:
                continue

        if temp_df is None or temp_df.empty:
            st.error("Unable to parse CSV file. Ensure it has at least 2 columns.")
            st.stop()

        columns = temp_df.columns.tolist()

        # Data preview
        st.subheader("Data Preview")
        st.write(f"Columns detected: {', '.join(columns)}")
        st.dataframe(temp_df.head(), use_container_width=True)

        # Column selection with better UI
        st.subheader("🎯 Column Selection")

        # Auto-detect columns
        date_col = next((col for col in columns if any(keyword in str(col).lower()
                        for keyword in ['date', 'time', 'timestamp'])), columns[0])

        # Find numeric value column
        preferred_keywords = ['confirmed', 'cases', 'deaths', 'recovered', 'value', 'price', 'close', 'gdp', 'sales']
        value_col = None

        # Try preferred columns first
        for col in columns:
            if col != date_col and any(keyword in str(col).lower() for keyword in preferred_keywords):
                if pd.api.types.is_numeric_dtype(temp_df[col]) or temp_df[col].dropna().astype(str).str.replace(',', '').str.replace('.', '').str.isnumeric().all():
                    value_col = col
                    break

        # Fallback to any numeric column
        if value_col is None:
            for col in columns:
                if col != date_col and pd.api.types.is_numeric_dtype(temp_df[col]):
                    value_col = col
                    break

        if value_col is None and len(columns) > 1:
            value_col = columns[1]

        if value_col is None:
            st.error("❌ No suitable numeric value column found.")
            st.stop()

        # Display selection
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Date Column: {date_col}")
        with col2:
            st.info(f"Value Column: {value_col}")

        # Manual override option
        with st.expander("Manual Column Selection"):
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select Date Column", columns, index=columns.index(date_col))
            with col2:
                value_options = [c for c in columns if c != date_col]
                value_col = st.selectbox("Select Value Column", value_options)

    except Exception as e:
        st.error(f"File processing error: {str(e)}")
        st.stop()
else:
    # Clear data when no file is uploaded
    if 'last_file_name' in st.session_state:
        del st.session_state.last_file_name
    st.session_state.results = None
    st.info("ℹ️ No file uploaded. Using synthetic financial data for demonstration.")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Forecasting section
st.subheader("Run Forecasting")

# Center the main buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.empty()
with col2:
    # Main action buttons side by side
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_button = st.button("🚀 Generate Forecast", use_container_width=True, type="primary")
    with btn_col2:
        clear_button = st.button("🗑️ Clear Results", use_container_width=True)
        if clear_button:
            st.session_state.results = None
            st.rerun()
with col3:
    if not uploaded_file:
        if st.button("📊 View Sample Data"):
            sample_df = generate_monthly_series()
            st.dataframe(sample_df.head(10), use_container_width=True)
    else:
        st.empty()

def run_pipeline(uploaded_df=None):
    """Execute the complete forecasting pipeline."""
    DATA_DIR = Path("data")
    RESULTS = Path("results")
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # Data preparation
    if uploaded_df is not None:
        df = uploaded_df
        csv_path = None
    else:
        df = generate_monthly_series()
        csv_path = DATA_DIR / "synthetic_monthly.csv"
        df.to_csv(csv_path, index=False)

    # Process data
    prepared = load_and_prepare(df if csv_path is None else csv_path)
    prepared.to_csv(RESULTS / "prepared_series.csv")
    series = prepared['value']

    # Train/test split
    train, val, test = time_series_split(series)
    train.to_frame(name='value').reset_index().to_csv(RESULTS / "train.csv", index=False)
    val.to_frame(name='value').reset_index().to_csv(RESULTS / "val.csv", index=False)
    test.to_frame(name='value').reset_index().to_csv(RESULTS / "test.csv", index=False)

    # Model fitting
    fitted_res, best_order, best_seasonal = fit_sarimax_auto(train)
    if fitted_res is None:
        st.error("SARIMAX model fitting failed.")
        return

    with open(RESULTS / "model_summary.txt", "w") as f:
        f.write(f"Selected order: {best_order}, seasonal: {best_seasonal}\n\n")
        f.write(fitted_res.summary().as_text())

    # Forecasting
    n_periods = max(len(test), 12)
    preds, conf_int = forecast_sarimax(fitted_res, steps=n_periods)

    if preds.isna().any() or conf_int.isna().any().any():
        st.error("Invalid predictions generated. Try using more data or synthetic data.")
        return

    last_date = series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=n_periods, freq='M')
    forecasts_df = pd.DataFrame({
        'date': forecast_index,
        'forecast': preds.values,
        'ci_lower': conf_int.iloc[:,0].values,
        'ci_upper': conf_int.iloc[:,1].values
    })
    forecasts_df.to_csv(RESULTS / "forecasts.csv", index=False)

    # Evaluation
    eval_len = min(len(test), n_periods)
    y_true = test.values[:eval_len]
    y_pred = preds.values[:eval_len]
    metrics = evaluate(y_true, y_pred)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Generate narrative
    mean_forecast = float(preds.mean())
    hist_mean = float(series.mean())
    trend = "upward" if mean_forecast > hist_mean else "downward" if mean_forecast < hist_mean else "stable"
    narrative = generate_narrative(metrics, trend, model_info=f"order={best_order}, seasonal={best_seasonal}")
    (RESULTS / "narrative.txt").write_text(narrative)

    # Store results
    st.session_state.results = {
        'forecasts': forecasts_df,
        'metrics': metrics,
        'narrative': narrative,
        'series': series,
        'test': test
    }

if run_button:
    with st.spinner("Running SARIMAX forecasting pipeline..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Preparing data...")
        progress_bar.progress(20)

        if uploaded_file:
            selected_df = temp_df[[date_col, value_col]].copy()
            selected_df.columns = ['date', 'value']
            selected_df['date'] = pd.to_datetime(selected_df['date'], errors='coerce')
            selected_df = selected_df.dropna(subset=['date'])

            if selected_df.empty:
                st.error("No valid dates found in selected column.")
                st.stop()

            status_text.text("Training model...")
            progress_bar.progress(60)
            run_pipeline(selected_df)
        else:
            status_text.text("Training model...")
            progress_bar.progress(60)
            run_pipeline()

        status_text.text("Generating forecasts...")
        progress_bar.progress(90)

        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()

    st.success("Forecasting completed successfully.")

if st.session_state.results:
    # Results header
    st.markdown("""
    <div class="corporate-header">
        <h2 style="color: #333; margin: 0; font-size: 1.5rem; font-weight: 600;">Analysis Results</h2>
        <p style="color: #666; margin: 0.25rem 0 0 0; font-size: 0.9rem;">
            SARIMAX forecasting completed
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Visualization section
    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    series = st.session_state.results['series']
    forecasts = st.session_state.results['forecasts']
    test = st.session_state.results['test']

    # Compact plotting with smaller elements
    ax.plot(series.index, series.values, label='Historical', color='#0066cc', linewidth=1.5)
    ax.plot(test.index, test.values, label='Test', color='#666666', marker='o', markersize=3, linewidth=1)
    ax.plot(forecasts['date'], forecasts['forecast'], label='Forecast', color='#333333', linestyle='--', linewidth=1.5)
    ax.fill_between(forecasts['date'], forecasts['ci_lower'], forecasts['ci_upper'],
                    color='#e0e0e0', alpha=0.4, label='CI')

    # Compact legend and labels
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title("Forecast", fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=7)

    # Tight layout for compactness
    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)

    # Performance metrics section
    st.subheader("📊 Model Performance Metrics")

    metrics = st.session_state.results['metrics']

    if any(pd.isna(v) for v in metrics.values()):
        st.error("⚠️ Model failed to generate reliable predictions. This may be due to insufficient data or irregular patterns in the time series.")
        st.info("💡 **Suggestions:** Try using more data points (minimum 24 recommended) or data with clearer seasonal patterns.")
    else:
        # Enhanced metrics display with explanations
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Mean Absolute Error",
                format_large_number(metrics['MAE']),
                help="Average absolute difference between predicted and actual values. Lower is better."
            )
            st.caption("Measures prediction accuracy in original units")

        with col2:
            st.metric(
                "Root Mean Square Error",
                format_large_number(metrics['RMSE']),
                help="Square root of average squared differences. Penalizes large errors more. Lower is better."
            )
            st.caption("More sensitive to outliers than MAE")

        with col3:
            st.metric(
                "Mean Absolute % Error",
                f"{metrics['MAPE']:.1f}%",
                help="Average percentage error. Shows relative accuracy. Lower is better."
            )
            st.caption("Percentage-based accuracy measure")

        # Performance assessment with detailed feedback
        mape = metrics['MAPE']
        st.markdown("### 📈 Performance Assessment")

        if mape < 5:
            st.success("🎯 **Exceptional Performance** - Highly accurate predictions with very low error rates.")
            st.info("✅ This model demonstrates excellent forecasting capability for this dataset.")
        elif mape < 10:
            st.success("✅ **Excellent Performance** - Very accurate predictions with minimal errors.")
            st.info("✅ Model performs well and can be confidently used for forecasting.")
        elif mape < 15:
            st.info("👍 **Good Performance** - Reliable predictions with acceptable error levels.")
            st.info("✅ Model provides useful forecasts but consider fine-tuning for better accuracy.")
        elif mape < 25:
            st.warning("⚠️ **Fair Performance** - Predictions have moderate errors.")
            st.info("🔄 Consider using more data, different model parameters, or preprocessing techniques.")
        elif mape < 50:
            st.warning("⚠️ **Poor Performance** - High error rates may limit practical use.")
            st.info("🔄 Significant improvements needed. Try different data or model approaches.")
        else:
            st.error("❌ **Very Poor Performance** - Predictions are unreliable.")
            st.info("🔄 Model may not be suitable for this data. Consider alternative approaches or more data.")

        # Additional context
        st.markdown("### 📋 Understanding These Metrics")
        st.markdown("""
        - **MAE**: Shows average prediction error in the same units as your data
        - **RMSE**: Similar to MAE but penalizes large errors more heavily
        - **MAPE**: Shows percentage error, useful for comparing across different scales
        - **Lower values** = Better accuracy for all metrics
        """)

    # Summary section
    st.subheader("📋 Analysis Summary")

    with st.expander("🔍 Detailed Analysis Report", expanded=False):
        narrative = st.session_state.results['narrative']

        # Format the narrative into clear sections
        sections = [line.strip() for line in narrative.split('\n') if line.strip()]

        # Executive Summary
        st.markdown("### 📊 Executive Summary")
        if sections:
            st.info(f"**Key Finding:** {sections[0]}")
            if len(sections) > 1:
                st.write(f"**Trend Analysis:** {sections[1]}")

        # Model Details
        st.markdown("### 🤖 Model Configuration")
        model_info = f"SARIMAX model with automated parameter selection"
        st.info(model_info)

        # Performance Insights
        st.markdown("### 🎯 Performance Insights")
        mape = st.session_state.results['metrics']['MAPE']
        if mape < 15:
            st.success("✅ **Strong Performance** - Model predictions are reliable for decision making")
        elif mape < 30:
            st.warning("⚠️ **Moderate Performance** - Use predictions with caution and consider improvements")
        else:
            st.error("❌ **Weak Performance** - Consider alternative models or more data")

        # Recommendations
        st.markdown("### 💡 Recommendations")
        if mape < 15:
            st.info("• Model is ready for production use\n• Monitor performance quarterly\n• Consider ensemble methods for enhanced accuracy")
        elif mape < 30:
            st.info("• Gather more historical data if possible\n• Consider feature engineering\n• Evaluate alternative forecasting methods")
        else:
            st.info("• Significant model improvement needed\n• Consider consulting domain experts\n• Evaluate if forecasting is appropriate for this data")

        # Technical Details
        st.markdown("### 📈 Technical Details")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Points Analyzed", f"{len(st.session_state.results['series']):,}")
            st.metric("Training Data", f"{len(st.session_state.results['series']) * 0.7:.0f}")
        with col2:
            st.metric("Test Data", f"{len(st.session_state.results['test']):,}")
            st.metric("Forecast Horizon", f"{len(st.session_state.results['forecasts'])} periods")

        # Raw narrative
        st.markdown("### 📄 Raw Model Report")
        st.text_area("Model Details", narrative, height=200)