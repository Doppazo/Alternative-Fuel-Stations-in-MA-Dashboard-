import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import math
import os


# AI Assisted Page Configuration
st.set_page_config(
    page_title="MA Alternative Fuel Stations",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117; /* matches Streamlit's top bar */
        }
    </style>
    """,
    unsafe_allow_html=True)

st.markdown("""
    <style>
    html, body, [class*="css"] {
        color: #FF7F50 !important;
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #FF7F50 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FF7F50 !important;
    }

    .stDataFrame {
        color: #FF7F50 !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem;
    }

    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)


# Data Cleaning
def load_and_clean_data(filename='alt_fuel_stations.csv'):
    try:
        # Try different possible locations for the CSV file
        if os.path.exists('alt_fuel_stations.csv'):
            df = pd.read_csv('alt_fuel_stations.csv')
        elif os.path.exists('../alt_fuel_stations.csv'):
            df = pd.read_csv('../alt_fuel_stations.csv')
        elif os.path.exists('Final Project/alt_fuel_stations.csv'):
            df = pd.read_csv('Final Project/alt_fuel_stations.csv')
        else:
            st.error("Error: CSV file not found. Please check your data file location.")
            return None

        essential_columns = [
            'Station Name',
            'Street Address',
            'City',
            'State',
            'ZIP',
            'Fuel Type Code',
            'Access Code',
            'Latitude',
            'Longitude',
            'Access Days Time',
            'Date Last Confirmed',
            'Open Date',
            'Owner Type Code',
            'EV Connector Types',
            'EV Network',
        ]

        available_columns = [col for col in essential_columns if col in df.columns]
        df = df[available_columns]

        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df = df.dropna(subset=['Latitude', 'Longitude'])

        if 'City' in df.columns:
            df['City'] = df['City'].fillna('Unknown')

        if 'ZIP' in df.columns:
            df['ZIP'] = df['ZIP'].fillna('00000')
            df['ZIP'] = df['ZIP'].astype(str).str.split('.').str[0].str.zfill(5)

        if 'Access Code' in df.columns:
            df['Access Code'] = df['Access Code'].fillna('unknown')

        if 'Fuel Type Code' in df.columns:
            df['Fuel Type Code'] = df['Fuel Type Code'].str.strip()

        df = df.drop_duplicates()

        if 'Access Days Time' in df.columns:
            df['Is_24_Hour'] = df['Access Days Time'].str.contains('24 hours', case=False, na=False)

        if 'Date Last Confirmed' in df.columns:
            df['Date Last Confirmed'] = pd.to_datetime(df['Date Last Confirmed'])
            df['Year_Confirmed'] = df['Date Last Confirmed'].dt.year

        if 'Open Date' in df.columns:
            df['Open Date'] = pd.to_datetime(df['Open Date'], errors='coerce')
            df['Year_Opened'] = df['Open Date'].dt.year

        if all(col in df.columns for col in ['Street Address', 'City', 'State', 'ZIP']):
            df['Full_Address'] = df['Street Address'] + ', ' + df['City'] + ', ' + df['State'] + ' ' + df['ZIP']

        df = df.reset_index(drop=True)
        return df

    except FileNotFoundError:
        st.error(f"Error: File '{filename}' not found in the current directory.")
        return None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def data_summary(df):
    total_stations = len(df)
    num_cities = df["City"].nunique()
    return total_stations, num_cities


# AI Assisted Tab Design Functions
def show_table_tab(filtered_df, total_count):
    st.header("Station Data Table")

    total_stations, num_cities = data_summary(filtered_df)
    st.caption(f"Filtered stations: {total_stations} across {num_cities} cities")

    st.dataframe(filtered_df, use_container_width=True)
    st.info(f"Showing {len(filtered_df)} of {total_count} stations")

    st.download_button(
        label="Download filtered data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_stations.csv",
        mime="text/csv")


def show_map_tab(df):
    map_type = st.radio("Select Map Type:", ["Scatter Plot", "Heatmap"], horizontal=True)

    if map_type == "Scatter Plot":
        st.subheader("Station Locations")
        latitude_longitude(df)
    else:
        st.subheader("Station Density (Heatmap)")
        heatmap_stations(df)


def show_graph_tab(df):
    st.header("Station Analytics")
    chart_fueltype(df)
    st.markdown("---")
    chart_stations_per_fuel_type(df)
    st.markdown("---")
    chart_top_zipcodes(df)
    st.markdown("---")
    chart_fuel_types_by_city_stacked(df)
    st.markdown("---")
    chart_company_comparison(df)
    st.markdown("---")
    chart_fuel_types_for_city(df)
    st.markdown("---")
    chart_stations_over_time(df)


# Analysis Functions
def chart_fueltype(data):
    df = data

    unknown = []
    cng = []
    lpg = []
    elec = []
    e = []
    bd = []

    for index, row in df.iterrows():
        fuel_type = row['Fuel Type Code']
        station_name = row['Station Name']

        if fuel_type == '':
            unknown.append(station_name)
        elif fuel_type == 'CNG':
            cng.append(station_name)
        elif fuel_type == 'LPG':
            lpg.append(station_name)
        elif fuel_type == 'ELEC':
            elec.append(station_name)
        elif fuel_type == 'E85':
            e.append(station_name)
        elif fuel_type == 'BD':
            bd.append(station_name)

    len1 = len(unknown)
    len2 = len(cng)
    len3 = len(lpg)
    len4 = len(elec)
    len5 = len(e)
    len6 = len(bd)

    total_stationsbytype = len1 + len2 + len3 + len4 + len5 + len6

    unknown_percentage = (len1 / total_stationsbytype) * 100
    cng_percentage = (len2 / total_stationsbytype) * 100
    lpg_percentage = (len3 / total_stationsbytype) * 100
    elec_percentage = (len4 / total_stationsbytype) * 100
    e_percentage = (len5 / total_stationsbytype) * 100
    bd_percentage = (len6 / total_stationsbytype) * 100

    labels = ['Unknown', 'CNG', 'LPG', 'ELEC', 'E85', 'BD']
    values = [unknown_percentage, cng_percentage, lpg_percentage, elec_percentage, e_percentage, bd_percentage]
    counts = [len1, len2, len3, len4, len5, len6]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    explode = (0.05, 0, 0, 0, 0, 0)

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct='',
        startangle=90,
        explode=explode,
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
    )

    ax.set_title('Fuel Type Distribution by Station',
                 color='white',
                 fontsize=14,
                 weight='bold',
                 pad=15)

    legend_labels = [f'{label}: {val:.1f}% ({count} stations)'
                     for label, val, count in zip(labels, values, counts)]

    ax.legend(
        wedges,
        legend_labels,
        title="Fuel Types",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
        title_fontsize=11,
        frameon=True,
        facecolor='#262730',
        edgecolor='white',
        labelcolor='white'
    )

    plt.tight_layout()
    st.pyplot(fig)


def chart_stations_over_time(data):
    df = data
    date_station_count = {}

    for index, row in df.iterrows():
        date = row['Open Date']

        if pd.isna(date):
            continue

        if date in date_station_count:
            date_station_count[date] = date_station_count[date] + 1

        else:
            date_station_count[date] = 1

    if len(date_station_count) == 0:
        st.warning("No date confirmation data available for timeline chart.")
        return

    sorted_dates = sorted(date_station_count.keys())

    cumulative_totals = []
    running_total = 0

    for date in sorted_dates:
        running_total = running_total + date_station_count[date]
        cumulative_totals.append(running_total)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    ax.plot(sorted_dates,
            cumulative_totals,
            color='#4ECDC4',
            linewidth=2.5,
            marker='o',
            markersize=4)

    ax.set_title('Cumulative Growth of Alternative Fuel Stations Over Time',
                 color='white',
                 fontsize=14,
                 weight='bold',
                 pad=15)

    ax.set_xlabel('Date', color='white', fontsize=11)
    ax.set_ylabel('Total Number of Stations', color='white', fontsize=11)

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, alpha=0.2, color='white', linestyle='--')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    st.pyplot(fig)


def chart_stations_per_fuel_type(data):
    df = data

    fuel_counts = build_fuel_type_counts(df)

    labels = [f"{fuel} ({count})" for fuel, count in fuel_counts.items()]
    values = [count for _, count in fuel_counts.items()]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    ax.bar(labels, values, color='#4ECDC4')

    ax.set_title('Number of Stations per Fuel Type', color='white', fontsize=14, weight='bold')
    ax.set_xlabel('Fuel Type', color='white')
    ax.set_ylabel('Number of Stations', color='white')

    ax.tick_params(colors='white', rotation=30, axis='x')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)


def build_fuel_type_counts(df):
    fuel_counts = {}

    for index, row in df.iterrows():
        fuel = row["Fuel Type Code"]
        if type(fuel) != str or fuel.strip() == "":
            fuel = "Unknown"
        fuel = fuel.strip()

        if fuel in fuel_counts:
            fuel_counts[fuel] = fuel_counts[fuel] + 1
        else:
            fuel_counts[fuel] = 1

    return fuel_counts


def chart_top_zipcodes(data, top_n=15):
    df = data

    zip_counts = df.groupby('ZIP')['Station Name'].count().sort_values(ascending=False)
    top_zip = zip_counts.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    ax.barh(top_zip.index.astype(str), top_zip.values, color='#96CEB4')
    ax.invert_yaxis()

    ax.set_title(f'Top {top_n} ZIP Codes by Number of Stations', color='white', fontsize=14, weight='bold')
    ax.set_xlabel('Number of Stations', color='white')
    ax.set_ylabel('ZIP Code', color='white')

    ax.tick_params(colors='white')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)


def chart_fuel_types_by_city_stacked(df, top_n=10):
    if df.empty:
        st.warning("No data available for city fuel type chart.")
        return

    top_cities = df['City'].value_counts().head(top_n).index
    filtered = df[df['City'].isin(top_cities)]

    pivot = pd.pivot_table(
        filtered,
        index='City',
        columns='Fuel Type Code',
        values='Station Name',
        aggfunc='count',
        fill_value=0
    )

    pivot = pivot.assign(total=pivot.sum(axis=1)).sort_values('total', ascending=False)
    pivot = pivot.drop(columns='total')

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    bottom = None
    colors = plt.cm.Set3(range(len(pivot.columns)))

    for i, fuel_type in enumerate(pivot.columns):
        values = pivot[fuel_type].values
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=fuel_type,
            color=colors[i]
        )
        bottom = values if bottom is None else bottom + values

    ax.set_title(f"Fuel Type Distribution by City (Top {top_n} Cities)",
                 color='white', fontsize=14, weight='bold')
    ax.set_xlabel("City", color='white')
    ax.set_ylabel("Number of Stations", color='white')
    ax.tick_params(colors='white', rotation=45)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        title="Fuel Type",
        facecolor="#262730",
        edgecolor="white",
        labelcolor="white"
    )
    plt.tight_layout()
    st.pyplot(fig)


def chart_company_comparison(df, top_n=10):
    if 'EV Network' not in df.columns:
        st.info("EV Network data not available for company comparison.")
        return

    temp = df.copy()
    temp['EV Network'] = temp['EV Network'].fillna("Unknown")
    network_counts = temp['EV Network'].value_counts().head(top_n)

    if network_counts.empty:
        st.warning("No data available for company comparison.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    ax.bar(network_counts.index, network_counts.values, color='#96CEB4')

    ax.set_title(f"Top {top_n} Networks by Number of Stations",
                 color='white', fontsize=14, weight='bold')
    ax.set_xlabel("EV Network / Company", color='white')
    ax.set_ylabel("Number of Stations", color='white')
    ax.tick_params(colors='white', rotation=30, axis='x')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)


def chart_fuel_types_for_city(df):
    if df.empty:
        st.warning("No data available for city fuel type breakdown.")
        return

    cities = sorted(df['City'].dropna().unique().tolist())
    selected_city = st.selectbox("Select a city for fuel type breakdown", cities)

    city_df = df[df['City'] == selected_city]
    if city_df.empty:
        st.warning(f"No data available for {selected_city}.")
        return

    fuel_counts = (
        city_df['Fuel Type Code']
        .fillna("Unknown")
        .replace("", "Unknown")
        .value_counts()
    )


    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')

    ax.bar(fuel_counts.index, fuel_counts.values, color='#4ECDC4')

    ax.set_title(f"Fuel Types in {selected_city}", color='white', fontsize=14, weight='bold')
    ax.set_xlabel("Fuel Type", color='white')
    ax.set_ylabel("Number of Stations", color='white')
    ax.tick_params(colors='white', rotation=30, axis='x')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)


def latitude_longitude(data):
    df = data
    locations_dict = {}

    for index, row in df.iterrows():
        station_name = row['Station Name']
        latitude = row['Latitude']
        longitude = row['Longitude']

        locations_dict[station_name] = latitude, longitude

    locations = pd.DataFrame([
        {'station_name': name, 'lat': coords[0], 'lon': coords[1]}
        for name, coords in locations_dict.items()])


    layer = pdk.Layer(
        "ScatterplotLayer",
        locations,
        get_position=["lon", "lat"],
        get_radius=500,
        get_fill_color=[255, 0, 0, 160],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=locations['lat'].mean(),
        longitude=locations['lon'].mean(),
        zoom=9,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{station_name}"}
    ))


def heatmap_stations(data):
    df = data
    locations_dict = {}

    for index, row in df.iterrows():
        station_name = row['Station Name']
        latitude = row['Latitude']
        longitude = row['Longitude']

        locations_dict[station_name] = latitude, longitude

    locations = pd.DataFrame([
        {'station_name': name, 'lat': coords[0], 'lon': coords[1]}
        for name, coords in locations_dict.items()])

    layer = pdk.Layer(
        "HeatmapLayer",
        locations,
        get_position=["lon", "lat"],
        aggregation='"MEAN"',
        opacity=0.9,
        threshold=0.05,
    )

    view_state = pdk.ViewState(
        latitude=locations['lat'].mean(),
        longitude=locations['lon'].mean(),
        zoom=9,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10"
    ))


def distance_radius(lat1, lon1, lat2, lon2, radius_km=6371):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius_km * c


def apply_filters(df, search_text, fuel_type, only_24_hour, max_radius_km):
    filtered = df.copy()

    if search_text:
        filtered = filtered[filtered['Station Name'].str.contains(search_text, case=False, na=False)]

    if fuel_type != 'All':
        filtered = filtered[filtered['Fuel Type Code'] == fuel_type]

    if only_24_hour:
        filtered = filtered[filtered['Is_24_Hour'] == True]

    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()

    filtered['Distance_km'] = filtered.apply(
        lambda row: distance_radius(center_lat, center_lon, row['Latitude'], row['Longitude']),
        axis=1
    )

    if max_radius_km is not None:
        filtered = filtered[filtered['Distance_km'] <= max_radius_km]

    return filtered


# Main Function
def main():
    st.title("⛽ MA Alternative Fuel Stations Dashboard")

    df = load_and_clean_data()

    if df is not None:
        st.sidebar.header("Filters")
        search_text = st.sidebar.text_input("Search station name", "")

        fuel_types = ['All'] + sorted(df['Fuel Type Code'].dropna().unique().tolist())
        fuel_type = st.sidebar.selectbox("Filter by fuel type", fuel_types)

        only_24_hour = st.sidebar.checkbox("Show only 24-hour stations", value=False)

        max_radius_km = st.sidebar.slider(
            "Maximum distance from center (km)",
            min_value=0,
            max_value=100,
            value=100,
            step=5
        )
        if max_radius_km == 100:
            max_radius_km = None

        filtered_df = apply_filters(df, search_text, fuel_type, only_24_hour, max_radius_km)
        total_count = len(df)

        total_stations, num_cities = data_summary(df)
        st.sidebar.metric("Total Stations", total_stations)
        st.sidebar.metric("Cities Covered", num_cities)

        tab1, tab2, tab3 = st.tabs(["📋 Table View", "🗺️ Map View", "📊 Analytics"])

        with tab1:
            show_table_tab(filtered_df, total_count)

        with tab2:
            show_map_tab(filtered_df)

        with tab3:
            show_graph_tab(filtered_df)
    else:
        st.error("Unable to load data. Please check your data file.")


if __name__ == "__main__":
    main()
