import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

days_df = pd.read_csv("day.csv")
hours_df = pd.read_csv("hour.csv")

st.title("🚲 Bike Sharing 🚲")
st.write("Lets Sharing a Bike!")

def create_sum_casual_user_df(df):
    sum_casual_user_df = df.groupby("dteday").casual.sum().sort_values(ascending=False).reset_index()
    return sum_casual_user_df

def create_sum_registered_user_df(df):
    sum_registered_user_df = df.groupby("dteday").registered.sum().sort_values(ascending=False).reset_index()
    return sum_registered_user_df

datetime_columns = ["dteday"]
days_df.sort_values(by="dteday", inplace=True)
days_df.reset_index(inplace=True)

for column in datetime_columns:
    days_df[column] = pd.to_datetime(days_df[column])

min_date = days_df["dteday"].min()
max_date = days_df["dteday"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = days_df[(days_df["dteday"] >= str(start_date)) & 
                (days_df["dteday"] <= str(end_date))]

sum_casual_user_df = create_sum_casual_user_df(main_df)
sum_registeres_user_df = create_sum_registered_user_df(main_df)

st.header('Bike Sharing Dashboard :sparkles:')
st.subheader('Daily User')

col1, col2 = st.columns(2)

with col1:
    total_casual = sum_casual_user_df.casual.sum()
    st.metric("Total Casual User", value=total_casual)

    chart_casual = days_df["casual"]
    st.line_chart(chart_casual)

with col2:
    total_registered = sum_registeres_user_df.registered.sum() 
    st.metric("Total Registered User", value=total_registered)

    chart_registered = days_df["registered"]
    st.line_chart(chart_registered)

st.subheader('Comparison of Casual and Registered Users')
fig, ax = plt.subplots()
ax.plot(days_df['casual'], label='Casual', marker='o')
ax.plot(days_df['registered'], label='Registered', marker='o')
ax.set_xlabel('Days')
ax.set_ylabel('Total Users')
ax.legend()
st.pyplot(fig)

st.subheader('Effect of Weather on Total Bike Sharing Users')
fig, ax = plt.subplots()
scatter = ax.scatter(days_df['weathersit'], days_df['cnt'], c=days_df['temp'], cmap='coolwarm', s=100)
ax.set_xlabel('Weather (1=Clear, 2=Cloudy, 3=Light Rain 4=Heavy Rain)')
ax.set_ylabel('Total Users')
fig.colorbar(scatter, ax=ax, label='Temperature')
st.pyplot(fig)

st.title("Scatter Plot Seasonal User")

option = st.selectbox(
    'Choose User Type:',
    ('Casual', 'Registered')
)

st.write(f'Scatter Plot {option} User vs Season')
fig, ax = plt.subplots()

if option == 'Casual':
    ax.scatter(days_df['casual'], days_df['season'], color='blue', label='Casual')
    ax.set_xlabel('Total Casual User')
else:
    ax.scatter(days_df['registered'], days_df['season'], color='magenta', label='Registered')
    ax.set_xlabel('Total Registered User')

ax.set_ylabel('Season')
ax.legend()

st.pyplot(fig)

st.caption('Copyright © Seradevi')
