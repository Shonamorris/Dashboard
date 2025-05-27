import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os

# Streamlit page setup
st.set_page_config(page_title="Milton Keynes Dashboard", layout="wide")
st.title("Milton Keynes Crime Data and ONS Happiness Index (2022–2025)")
st.subheader("This Dashboard identifies trends between the 2022–2024 Happiness Index and Milton Keynes Crime Rates.")

# Load & Clean Crime Data
@st.cache_data
def load_and_clean_crime_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    cleaned_data = []

    for sheet in sheet_names:
        df = xls.parse(sheet)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)
        df.dropna(axis=1, how='all', inplace=True)
        columns_to_remove = ['crime_id', 'reported_by', 'falls_within', 'longitude', 'latitude',
                             'location', 'lsoa_code', 'lsoa_name', 'context']
        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
        df.rename(columns={'crime_type': 'category', 'last_outcome_category': 'outcome'}, inplace=True)

        if 'month' not in df.columns:
            continue

        columns_to_keep = ['month', 'category', 'outcome', 'street_name', 'town']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        df.dropna(inplace=True)

        cleaned_data.append(df)

    all_data = pd.concat(cleaned_data, ignore_index=True)
    all_data['month'] = pd.to_datetime(all_data['month'], errors='coerce')
    all_data.dropna(subset=['month'], inplace=True)
    all_data['year_month'] = all_data['month'].dt.to_period('M')

    return all_data

# Load & Clean Happiness Data
@st.cache_data
def load_and_prepare_happiness(file_path):
    df = pd.read_excel(file_path)
    df.columns = [
        "TimePeriod", "MeanHappiness",
        "LowHappinessPct", "LowHappinessAccuracy", 
        "MedHappinessPct", "MedHappinessAccuracy", 
        "HighHappinessPct", "HighHappinessAccuracy", 
        "VeryHighHappinessPct", "VeryHighHappinessAccuracy"
    ]
    df = df.drop(columns=[col for col in df.columns if "Accuracy" in col])

    existing = df['TimePeriod'].tolist()
    full_range = [f"Q{q} {y}" for y in range(2022, 2026) for q in range(1, 5)]
    missing = [q for q in full_range if q not in existing]

    future_rows = pd.DataFrame({
        'TimePeriod': missing,
        'MeanHappiness': [np.nan]*len(missing),
        'LowHappinessPct': [np.nan]*len(missing),
        'MedHappinessPct': [np.nan]*len(missing),
        'HighHappinessPct': [np.nan]*len(missing),
        'VeryHighHappinessPct': [np.nan]*len(missing)
    })

    df = pd.concat([df, future_rows], ignore_index=True)

    df['Year'] = df['TimePeriod'].str.extract(r'(\d{4})').astype(int)
    df['Quarter'] = df['TimePeriod'].str.extract(r'Q([1-4])').astype(int)
    df = df.sort_values(by=['Year', 'Quarter']).reset_index(drop=True)
    df['TimeIndex'] = np.arange(len(df))

    return df

# File paths – change these to relative if uploading to the cloud
crime_file = "Milton Keynes Data 2024-2022.xlsx"
happy_file = "Happiness 2022 - 2024.xlsx"
image_path = "MK.png"

# Load Data
crime_df = load_and_clean_crime_data(crime_file)
happy_df = load_and_prepare_happiness(happy_file)

# Define Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Welcome",
    "📊 Crime Data",
    "📈 Crime Outcomes",
    "🚨 Violent Crime Focus",
    "😊 Happiness & Forecasting"
])

# ----------- TAB 0: Welcome -------------
with tab0:
    st.markdown("<h1 style='text-align: center; color: darkred;'> Milton Keynes Crime Data and ONS Happiness Index Dashboard</h1>", unsafe_allow_html=True)
    st.image(image_path, use_container_width=True)

    st.markdown("""
    <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
    This interactive dashboard explores the trends in Crime Data from Milton Keynes alongside the national ONS Happiness Index.  
    The goal is to identify trends and insights between local crime reports and national well-being from 2022 to 2025.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    **Student Code:** 100353764  
    **Tools Used:** Streamlit, Python, Pandas, Seaborn, Scikit-learn  
    **Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
    """)

# ----------- TAB 1: Crime Data -------------
with tab1:
    st.header("Crime in Milton Keynes (2022–2024)")

    total_crimes = len(crime_df)
    crimes_by_category = crime_df['category'].value_counts()
    most_common_category = crimes_by_category.idxmax()
    most_common_category_count = crimes_by_category.max()
    unique_months = crime_df['year_month'].nunique()
    avg_crimes_per_month = total_crimes / unique_months if unique_months else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Crimes Reported", f"{total_crimes:,}")
    col2.metric("Unique Months Recorded", unique_months)
    col3.metric("Avg Crimes/Month", f"{avg_crimes_per_month:.1f}")
    col4.metric("Top Crime Category", f"{most_common_category} ({most_common_category_count})")

    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    crimes_by_category.plot.pie(autopct='%1.1f%%', startangle=140, ax=ax_pie,
                                colors=sns.color_palette("Reds", len(crimes_by_category)))
    ax_pie.set_ylabel('')
    ax_pie.set_title("Crime Category Proportions (2022–2024)")
    st.pyplot(fig_pie)

    monthly_totals = crime_df.groupby('year_month').size()
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    monthly_totals.plot(kind='bar', color='firebrick', ax=ax1)
    ax1.set_title('Total Crimes Reported Each Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Crimes')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    cat_trends = crime_df.groupby(['year_month', 'category']).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    for col in cat_trends.columns:
        ax2.plot(cat_trends.index.to_timestamp(), cat_trends[col], label=col)
    ax2.set_title('Monthly Crime Trends by Category')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Crimes')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)

    st.subheader("Distribution of Monthly Crimes by Category")
    monthly_by_category = crime_df.groupby(['category', 'year_month']).size().reset_index(name='monthly_count')
    fig6, ax6 = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=monthly_by_category, x='category', y='monthly_count', color='firebrick', ax=ax6)
    ax6.set_title("Monthly Crime Distribution by Category")
    ax6.set_xlabel("Crime Category")
    ax6.set_ylabel("Monthly Crime Count")
    ax6.tick_params(axis='x', rotation=45)
    st.pyplot(fig6)

# ----------- TAB 2: Crime Outcomes -------------
with tab2:
    st.header("Crime Outcomes Over Time")

    outcome_trends = crime_df.groupby(['year_month', 'outcome']).size().unstack(fill_value=0)

    fig_outcome, ax_outcome = plt.subplots(figsize=(16, 8))
    for col in outcome_trends.columns:
        ax_outcome.plot(outcome_trends.index.to_timestamp(), outcome_trends[col], label=col)
    ax_outcome.set_title('Monthly Crime Outcomes')
    ax_outcome.set_xlabel('Month')
    ax_outcome.set_ylabel('Number of Outcomes')
    ax_outcome.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_outcome.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_outcome)

# ----------- TAB 3: Violent Crime -------------
with tab3:
    st.header("Violence and Sexual Offences in Milton Keynes (2022–2024)")

    violent = crime_df[crime_df['category'].str.lower() == 'violence and sexual offences']
    violent_outcomes = violent.groupby('outcome').size().sort_values(ascending=False)

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    violent_outcomes.plot(kind='bar', color='firebrick', ax=ax4)
    ax4.set_title('Outcomes for Violence and Sexual Offences')
    ax4.set_xlabel('Outcome')
    ax4.set_ylabel('Number of Crimes')
    ax4.tick_params(axis='x', rotation=45)
    fig4.tight_layout()
    st.pyplot(fig4)

    violent_trend = violent.groupby('year_month').size()
    fig5, ax5 = plt.subplots(figsize=(14, 6))
    violent_trend.plot(kind='line', marker='o', color='firebrick', ax=ax5)
    ax5.set_title('Monthly Trend of Violence and Sexual Offences')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Number of Crimes')
    ax5.grid(True, linestyle='--', alpha=0.6)
    fig5.tight_layout()
    st.pyplot(fig5)

# ----------- TAB 4: Happiness Forecasting -------------
with tab4:
    st.header("Happiness Trends and Forecast (2022–2025)")

    actual = happy_df[happy_df['MeanHappiness'].notna()]
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    ax7.plot(actual['TimePeriod'], actual['MeanHappiness'], marker='o')
    ax7.set_title("Mean Happiness Over Time")
    ax7.set_xlabel("Time Period")
    ax7.set_ylabel("Happiness Score (0–10)")
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True)
    st.pyplot(fig7)

    happy_df['Season'] = happy_df['TimePeriod'].apply(
        lambda q: 'Winter' if 'Q1' in q else 'Spring' if 'Q2' in q else 'Summer' if 'Q3' in q else 'Autumn'
    )

    selected_seasons = st.multiselect("Select Seasons to Display", ['Winter', 'Spring', 'Summer', 'Autumn'], default=['Winter', 'Spring', 'Summer', 'Autumn'])
    seasonal = happy_df.groupby('Season')['MeanHappiness'].mean().reindex(selected_seasons)
    fig8, ax8 = plt.subplots(figsize=(8, 5))
    seasonal.plot(kind='bar', color='lightgreen', ax=ax8)
    ax8.set_title("Average Happiness Score by Season")
    ax8.set_ylim(0, 10)
    ax8.set_ylabel("Mean Score")
    ax8.grid(axis='y')
    st.pyplot(fig8)

    model = LinearRegression()
    train_df = happy_df[happy_df['MeanHappiness'].notna()]
    model.fit(train_df[['TimeIndex']], train_df['MeanHappiness'])
    r2 = model.score(train_df[['TimeIndex']], train_df['MeanHappiness'])
    pred_train = model.predict(train_df[['TimeIndex']])
    rmse = np.sqrt(mean_squared_error(train_df['MeanHappiness'], pred_train))

    st.markdown("### Regression Model Performance")
    st.markdown(f"- **R²:** {r2:.3f}")
    st.markdown(f"- **RMSE:** {rmse:.3f}")

    happy_df['PredictedHappiness'] = model.predict(happy_df[['TimeIndex']])
    happy_df['MeanHappinessFilled'] = happy_df['MeanHappiness'].combine_first(happy_df['PredictedHappiness'])

    last_train_idx = train_df['TimeIndex'].max()
    n_forecast = happy_df['MeanHappiness'].isna().sum()
    if n_forecast > 0:
        adjustment = np.linspace(0.1, 0.2, n_forecast)
        happy_df.loc[happy_df['MeanHappiness'].isna(), 'PredictedHappiness'] -= adjustment
        happy_df['MeanHappinessFilled'] = happy_df['MeanHappiness'].combine_first(happy_df['PredictedHappiness'])

    fig9, ax9 = plt.subplots(figsize=(12, 6))
    ax9.plot(happy_df['TimePeriod'], happy_df['MeanHappinessFilled'], marker='o', label='Actual + Predicted')
    ax9.axvline(x=train_df.iloc[-1]['TimePeriod'], color='red', linestyle='--', label='Prediction Start')
    ax9.set_title("Forecast of Mean Happiness Scores (2022–2025)")
    ax9.set_xlabel("Time Period")
    ax9.set_ylabel("Happiness Score")
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True)
    ax9.legend()
    fig9.tight_layout()
    st.pyplot(fig9)

# Footer
st.markdown("---")
st.markdown("*Data Sources:* Milton Keynes Crime Data (2022–2024), ONS Happiness Index (2022–2024)")
st.write("Current working directory:", os.getcwd())
