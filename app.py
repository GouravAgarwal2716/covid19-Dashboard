import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Page config
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide", page_icon="🦠")

# Load and cache live data
@st.cache_data(ttl=3600)
def load_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df = df[df['iso_code'].str.len() == 3]  # Filter valid countries
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Title and description
st.title("🦠 COVID-19 Data Science Dashboard")
st.markdown("""
This interactive dashboard allows you to explore real-time COVID-19 metrics across countries.
You can compare countries, apply date ranges, see rolling averages, and even forecast future cases.
""")

# Sidebar filters
st.sidebar.header("🔎 Filters")
countries = st.sidebar.multiselect("🌍 Select Countries", sorted(df['location'].unique()), default=["India"])
metric = st.sidebar.selectbox("📊 Select Metric", ['new_cases', 'new_deaths', 'new_vaccinations'])

# Filter data
filtered_df = df[df['location'].isin(countries)]

# Date range filter
min_date = filtered_df['date'].min()
max_date = filtered_df['date'].max()
start_date, end_date = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
filtered_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(start_date)) & (filtered_df['date'] <= pd.to_datetime(end_date))]

# Optional smoothing
smooth = st.sidebar.checkbox("📈 Show 7-Day Rolling Average")
if smooth:
    filtered_df[metric] = filtered_df.groupby('location')[metric].transform(lambda x: x.rolling(7).mean())

# Drop NaNs after rolling
filtered_df = filtered_df.dropna(subset=[metric])

# Summary stats
st.subheader("📋 Summary Statistics")
for country in countries:
    st.markdown(f"**{country}**")
    st.write(filtered_df[filtered_df['location'] == country][metric].describe())

# Download button
csv = filtered_df[['location', 'date', metric]].to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=csv,
    file_name=f'covid_data_{metric}.csv',
    mime='text/csv'
)

# Main line chart
st.subheader(f"📈 {metric.replace('_', ' ').title()} Over Time")
fig = px.line(
    filtered_df,
    x='date',
    y=metric,
    color='location',
    labels={'date': 'Date', metric: metric.replace('_', ' ').title(), 'location': 'Country'},
    template='plotly_white'
)
fig.update_layout(xaxis_title='Date', yaxis_title=metric.replace('_', ' ').title(), legend_title='Country')
st.plotly_chart(fig, use_container_width=True)

# Forecasting section
if len(countries) == 1 and st.checkbox("🔮 Show 30-Day Forecast (Prophet)"):
    st.subheader(f"🔮 Forecast for {countries[0]} - {metric.replace('_', ' ').title()}")

    forecast_df = filtered_df[filtered_df['location'] == countries[0]][['date', metric]].dropna()
    forecast_df = forecast_df.rename(columns={'date': 'ds', metric: 'y'})

    if len(forecast_df) >= 30:
        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot forecast
        forecast_fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast for {countries[0]} ({metric.replace('_', ' ').title()})")
        forecast_fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Value")
        st.plotly_chart(forecast_fig, use_container_width=True)
    else:
        st.warning("Not enough data to forecast. Try selecting a longer date range or different country.")
