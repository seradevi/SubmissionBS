import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

days_df = pd.read_csv("dashboard/day.csv")
hours_df = pd.read_csv("dashboard/hour.csv")

st.title("🚲 Bike Sharing 🚲")
st.write("Lets Sharing a Bike!")

def create_sum_casual_user_df(df):
    sum_casual_user_df = df.groupby("dteday").casual.sum().sort_values(ascending=False).reset_index()
    return sum_casual_user_df

def create_sum_registered_user_df(df):
    sum_registered_user_df = df.groupby("dteday").registered.sum().sort_values(ascending=False).reset_index()
    return sum_registered_user_df

def seasonal_bike_sharing():
    day_df_filtered = days_df[days_df['yr'] == 1]
    seasonal_usage = day_df_filtered.groupby('season')[['casual', 'registered']].mean().reset_index()
    season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    seasonal_usage['season'] = seasonal_usage['season'].map(season_labels)

    st.subheader("Seasonal Bike Sharing (2012)")
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='value', hue='variable', data=pd.melt(seasonal_usage, ['season']))

    plt.title('Average Bicycle Usage by Casual and Registered Users Across Seasons (2012)', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Average Number of Users', fontsize=12)
    plt.legend(title='User Type', loc='upper right')
    plt.tight_layout()
    st.pyplot(plt)
    
def hourly_average_bike_sharing():
    st.subheader("Hourly Average Bike Sharing Per Day of the Week")

    hourly_avg_users = hours_df.groupby(['hr', 'weekday'])['cnt'].mean().unstack()
    plt.figure(figsize=(12, 6))
    for weekday in range(7):
        plt.plot(hourly_avg_users.index, hourly_avg_users[weekday], label=f'Weekday {weekday}')

    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Average Number of Users', fontsize=12)
    plt.title('Average Number of Users per Hour in a Week', fontsize=14)
    plt.legend(title='Day of the Week', loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def plot_weather_impact():
    last_year_data = days_df[days_df['yr'] == 1].copy()
    last_year_data.loc[:, 'day_type'] = last_year_data['weekday'].apply(lambda x: 'Weekend' if x == 0 or x == 6 else 'Working Day')

    working_day_data = last_year_data[last_year_data['day_type'] == 'Working Day']
    weekend_data = last_year_data[last_year_data['day_type'] == 'Weekend']
    working_corr_temp = working_day_data[['cnt', 'temp', 'hum']].corr().loc['cnt']
    weekend_corr_temp = weekend_data[['cnt', 'temp', 'hum']].corr().loc['cnt']

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Weather Impact on Total Bike Sharing (Working Days)')
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='temp', y='cnt', hue='hum', size='hum', sizes=(20, 200), data=working_day_data, palette="coolwarm")
        plt.title('Weather Impact on Total Bicycle Usage (Working Days)', fontsize=14)
        plt.xlabel('Temperature (Normalized)', fontsize=12)
        plt.ylabel('Total Users', fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)

    with col2:
        st.subheader('Weather Impact on Total Bike Sharing (Weekends)')
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='temp', y='cnt', hue='hum', size='hum', sizes=(20, 200), data=weekend_data, palette="coolwarm")
        plt.title('Weather Impact on Total Bicycle Usage (Weekends)', fontsize=14)
        plt.xlabel('Temperature (Normalized)', fontsize=12)
        plt.ylabel('Total Users', fontsize=12)
        plt.tight_layout()
        st.pyplot(plt)

def plot_weekday_weekend_usage(): 
    hours_df['is_weekend'] = hours_df['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)
    avg_bike_usage_by_hour = hours_df.groupby(['hr', 'is_weekend'])['cnt'].mean().reset_index()
    avg_bike_usage_by_hour_pivot = avg_bike_usage_by_hour.pivot(index='hr', columns='is_weekend', values='cnt')
    avg_bike_usage_by_hour_pivot.columns = ['Weekday', 'Weekend']
    
    st.subheader("Average Bike Usage per Hour: Weekdays vs Weekends")
    plt.figure(figsize=(10, 6))
    plt.plot(avg_bike_usage_by_hour_pivot.index, avg_bike_usage_by_hour_pivot['Weekday'], label='Weekday', marker='o')
    plt.plot(avg_bike_usage_by_hour_pivot.index, avg_bike_usage_by_hour_pivot['Weekend'], label='Weekend', marker='o', linestyle='--')
    plt.title('Average Bike Usage per Hour: Weekdays vs Weekends', fontsize=14)
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel('Average Number of Bikes Used', fontsize=12)
    plt.xticks(range(0, 24))  # Ensure all hours are shown on the x-axis
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def plot_holiday_usage():
    recent_data = days_df[days_df['yr'] == 1]
    holiday_vs_non_holiday = recent_data.groupby('holiday')[['casual', 'registered']].mean().reset_index()
    holiday_vs_non_holiday.columns = ['Holiday', 'Avg Casual Users', 'Avg Registered Users']
    
    st.subheader("Average Bike Sharing on Holidays vs Non-Holidays")
    plt.figure(figsize=(8, 6))
    bar_width = 0.35
    index = [0, 1]  
    plt.bar(index, holiday_vs_non_holiday['Avg Casual Users'], bar_width, label='Casual Users', color='skyblue')
    plt.bar([i + bar_width for i in index], holiday_vs_non_holiday['Avg Registered Users'], bar_width, label='Registered Users', color='orange')

    plt.title('Average Bike Usage on Holidays vs Non-Holidays', fontsize=14)
    plt.xlabel('Holiday (0: Non-Holiday, 1: Holiday)', fontsize=12)
    plt.ylabel('Average Number of Users', fontsize=12)
    plt.xticks([i + bar_width / 2 for i in index], ['Non-Holiday', 'Holiday'])
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

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

if __name__ == "__main__":
    hourly_average_bike_sharing()
    seasonal_bike_sharing()
    plot_weather_impact()
    plot_weekday_weekend_usage()
    plot_holiday_usage()

st.caption('Copyright © Seradevi')
