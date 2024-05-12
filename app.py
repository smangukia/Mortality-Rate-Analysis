import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Mortality Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #333333; /* Dark text color for better visibility */
    }
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .info-box {
            background-color: #262730;
            color: #ffffff;
            border: 1px solid #4e4e4e;
        }
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/merged_mortality_data.csv')
        return df
    except FileNotFoundError:
        st.warning("Original data file not found. Creating sample data for demonstration.")
        return create_sample_data()

def create_sample_data():
    countries = ['USA', 'CAN', 'JPN', 'GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'RUS', 'CHN']
    years = list(range(1990, 2022))
    
    data = []
    for country in countries:
        base = np.random.randint(8000, 12000)
        trend = np.random.uniform(1.0, 1.02)
        noise_level = np.random.uniform(0.05, 0.1)
        
        for i, year in enumerate(years):
            value = base * (trend ** i) * (1 + np.random.normal(0, noise_level))
            
            if year in [2001, 2009, 2020]:
                value *= np.random.uniform(1.1, 1.3) 
            
            data.append({
                'country': country,
                'year': year,
                'total': round(value),
                'sex': 1 
            })
            
            value_female = value * np.random.uniform(0.9, 1.1)
            data.append({
                'country': country,
                'year': year,
                'total': round(value_female),
                'sex': 2 
            })
    
    df = pd.DataFrame(data)
    
    age_groups = ['d0', 'd1', 'd5', 'd10', 'd15', 'd20', 'd25', 'd30', 'd35', 
                  'd40', 'd45', 'd50', 'd55', 'd60', 'd65', 'd70', 'd75', 'd80', 'd85p']
    
    for age in age_groups:
        if age in ['d0', 'd1']:
            df[age] = df['total'] * np.random.uniform(0.01, 0.03, len(df))
        elif age in ['d5', 'd10', 'd15', 'd20', 'd25', 'd30', 'd35']:
            df[age] = df['total'] * np.random.uniform(0.01, 0.05, len(df))
        elif age in ['d40', 'd45', 'd50', 'd55']:
            df[age] = df['total'] * np.random.uniform(0.05, 0.1, len(df))
        else:
            df[age] = df['total'] * np.random.uniform(0.1, 0.3, len(df))
    
    df.to_csv('data/merged_mortality_data.csv', index=False)
    return df

@st.cache_data
def detect_anomalies(df, method='zscore', threshold=2.0):

    result_df = df.copy()
    
    for country in df['country'].unique():
        for sex in df[df['country'] == country]['sex'].unique():
            mask = (df['country'] == country) & (df['sex'] == sex)
            country_data = df[mask].sort_values('year')
            
            if method == 'zscore':
                mean_val = country_data['total'].mean()
                std_val = country_data['total'].std()
                
                if std_val > 0: 
                    z_scores = (country_data['total'] - mean_val) / std_val
                    result_df.loc[country_data.index, 'anomaly_score'] = z_scores
                    result_df.loc[country_data.index, 'is_anomaly'] = (z_scores.abs() > threshold).astype(int)
            
            elif method == 'iqr':
                Q1 = country_data['total'].quantile(0.25)
                Q3 = country_data['total'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                result_df.loc[country_data.index, 'anomaly_score'] = (
                    (country_data['total'] - (Q1 + Q3) / 2) / IQR
                )
                result_df.loc[country_data.index, 'is_anomaly'] = (
                    (country_data['total'] < lower_bound) | 
                    (country_data['total'] > upper_bound)
                ).astype(int)
            
            elif method == 'moving_avg':
                if len(country_data) >= 5: 
                    country_data = country_data.sort_values('year')
                    rolling_mean = country_data['total'].rolling(window=5, min_periods=3, center=True).mean()
                    rolling_std = country_data['total'].rolling(window=5, min_periods=3, center=True).std()
                    
                    rolling_mean = rolling_mean.fillna(country_data['total'].mean())
                    rolling_std = rolling_std.fillna(country_data['total'].std())
                    
                    deviation = (country_data['total'] - rolling_mean) / rolling_std
                    
                    result_df.loc[country_data.index, 'anomaly_score'] = deviation
                    result_df.loc[country_data.index, 'is_anomaly'] = (deviation.abs() > threshold).astype(int)
    

    result_df['anomaly_score'] = result_df['anomaly_score'].fillna(0)
    result_df['is_anomaly'] = result_df['is_anomaly'].fillna(0)
    
    return result_df

@st.cache_data
def calculate_excess_mortality(df, window_size=5):
 
    result_df = df.copy()
    result_df['expected_mortality'] = np.nan
    result_df['excess_mortality'] = np.nan
    result_df['excess_percent'] = np.nan
    result_df['lower_bound'] = np.nan
    result_df['upper_bound'] = np.nan
    result_df['significant'] = 0
    
    for country in df['country'].unique():
        for sex in df[df['country'] == country]['sex'].unique():
            mask = (df['country'] == country) & (df['sex'] == sex)
            country_data = df[mask].sort_values('year')
            
            for i, (idx, row) in enumerate(country_data.iterrows()):
                if i >= window_size:
                    prev_years = country_data.iloc[i-window_size:i]
                    
                    mean_mortality = prev_years['total'].mean()
                    std_mortality = prev_years['total'].std()
                    
                    import scipy.stats as stats
                    t_value = stats.t.ppf(0.975, len(prev_years) - 1)
                    margin = t_value * std_mortality / np.sqrt(len(prev_years))
                    
                    lower_bound = mean_mortality - margin
                    upper_bound = mean_mortality + margin
                    
                    actual = row['total']
                    excess = actual - mean_mortality
                    excess_percent = (excess / mean_mortality) * 100
                    
                    significant = 1 if actual > upper_bound else 0
                    
                    result_df.loc[idx, 'expected_mortality'] = mean_mortality
                    result_df.loc[idx, 'excess_mortality'] = excess
                    result_df.loc[idx, 'excess_percent'] = excess_percent
                    result_df.loc[idx, 'lower_bound'] = lower_bound
                    result_df.loc[idx, 'upper_bound'] = upper_bound
                    result_df.loc[idx, 'significant'] = significant
    
    return result_df

@st.cache_data
def get_historical_events():
    return {
        1968: ["Hong Kong Flu Pandemic"],
        1969: ["Hong Kong Flu Pandemic aftermath"],
        1981: ["HIV/AIDS first identified"],
        2002: ["SARS outbreak begins"],
        2003: ["SARS outbreak continues"],
        2009: ["H1N1 (Swine Flu) Pandemic"],
        2014: ["Ebola outbreak in West Africa"],
        2018: ["Measles resurgence globally"],
        2019: ["Global measles outbreaks", "Samoa measles epidemic"],
        2020: ["COVID-19 Pandemic"],
        2021: ["COVID-19 Pandemic continues"],
        
        1991: ["Gulf War"],
        1992: ["Bosnian War begins"],
        1994: ["Rwandan Genocide"],
        1999: ["Kosovo War"],
        2001: ["9/11 Terrorist Attacks", "War in Afghanistan begins"],
        2003: ["Iraq War begins"],
        2011: ["Syrian Civil War begins", "Libyan Civil War"],
        2014: ["Crimean Crisis", "War in Eastern Ukraine begins"],
        2017: ["Yemen cholera outbreak"],
        
        1970: ["Bhola Cyclone in Bangladesh"],
        1976: ["Tangshan Earthquake in China"],
        1985: ["Mexico City Earthquake"],
        1995: ["Great Hanshin Earthquake in Japan"],
        2004: ["Indian Ocean Earthquake and Tsunami"],
        2005: ["Hurricane Katrina in USA"],
        2008: ["Cyclone Nargis in Myanmar", "Sichuan Earthquake in China"],
        2010: ["Haiti Earthquake"],
        2011: ["Tohoku Earthquake and Tsunami in Japan"],
        
        2003: ["European Heat Wave"],
        2010: ["Russian Heat Wave"],
        2015: ["Indian Heat Wave"],

        1990: ["German Reunification"],
        1991: ["Dissolution of USSR"],
        2010: ["Affordable Care Act in USA"]
    }

@st.cache_data
def match_with_historical_events(df):

    historical_events = get_historical_events()
    result_df = df.copy()
    
    result_df['matched_events'] = None
    result_df['has_matched_event'] = False
    result_df['event_type'] = 'unexplained'
    
    event_types = {
        'pandemic': {1968, 1969, 1970, 1981, 1982, 1983, 2002, 2003, 2009, 2010, 2014, 2015, 2018, 2019, 2020, 2021},
        'conflict': {1991, 1992, 1993, 1994, 1995, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2011, 2012, 2013, 2014, 2015},
        'natural_disaster': {1970, 1976, 1985, 1995, 2004, 2005, 2008, 2010, 2011},
        'policy_change': {1990, 1991, 2010},
        'heat_wave': {2003, 2010, 2015}
    }
    
    for idx, row in result_df[result_df['is_anomaly'] == 1].iterrows():
        year = row['year']
        
        if year in historical_events:
            result_df.loc[idx, 'matched_events'] = str(historical_events[year])
            result_df.loc[idx, 'has_matched_event'] = True
            
            for event_type, years in event_types.items():
                if year in years:
                    result_df.loc[idx, 'event_type'] = event_type
                    break
    
    return result_df

def main():
    # Sidebar
    # st.sidebar.image("mortalityrateanalysis.png", width=200)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Data Exploration", "Anomaly Detection", "Historical Correlation", "Excess Mortality"]
    )
    
    df = load_data()
    
    if page == "Home":
        st.markdown("<h1 class='main-header'>Mortality Analysis Dashboard</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p>This dashboard provides interactive tools to analyze mortality data across different countries and time periods. 
        It includes anomaly detection, historical correlation analysis, and excess mortality calculations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Countries", len(df['country'].unique()))
        
        with col2:
            st.metric("Years", f"{df['year'].min()} - {df['year'].max()}")
        
        with col3:
            st.metric("Total Records", len(df))
        
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)

        st.dataframe(df.head(10))
        
        st.markdown("<h2 class='sub-header'>Data Summary</h2>", unsafe_allow_html=True)

        country_summary = df.groupby('country').agg(
            min_year=('year', 'min'),
            max_year=('year', 'max'),
            avg_mortality=('total', 'mean'),
            min_mortality=('total', 'min'),
            max_mortality=('total', 'max')
        ).reset_index()
        
        numeric_cols = ['avg_mortality', 'min_mortality', 'max_mortality']
        country_summary[numeric_cols] = country_summary[numeric_cols].round(2)

        st.dataframe(country_summary)
        
        st.markdown("<h2 class='sub-header'>Global Mortality Trend</h2>", unsafe_allow_html=True)
        
        yearly_data = df.groupby('year')['total'].mean().reset_index()

        fig = px.line(
            yearly_data, 
            x='year', 
            y='total',
            title='Average Mortality Rate Over Time',
            labels={'total': 'Average Mortality', 'year': 'Year'},
            template='plotly_white'
        )
        
        historical_events = get_historical_events()
        event_years = []
        event_labels = []
        
        for year, events in historical_events.items():
            if year >= df['year'].min() and year <= df['year'].max():
                event_years.append(year)
                event_labels.append(events[0])
        
        if event_years:
            event_data = yearly_data[yearly_data['year'].isin(event_years)]
            
            fig.add_trace(
                go.Scatter(
                    x=event_data['year'],
                    y=event_data['total'],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Historical Events'
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h2 class='sub-header'>App Features</h2>", unsafe_allow_html=True)
        
        features = [
            ("Data Exploration", "Explore mortality data by country, year, and demographic factors."),
            ("Anomaly Detection", "Identify unusual mortality patterns using statistical methods."),
            ("Historical Correlation", "Correlate mortality anomalies with historical events."),
            ("Excess Mortality", "Calculate and visualize excess mortality during anomalous periods.")
        ]
        
        for title, description in features:
            st.markdown(f"**{title}**: {description}")
    
    elif page == "Data Exploration":
        st.markdown("<h1 class='main-header'>Data Exploration</h1>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Filters</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_countries = st.multiselect(
                "Select Countries",
                options=sorted(df['country'].unique()),
                default=sorted(df['country'].unique())[:3]
            )
        
        with col2:
            min_year, max_year = int(df['year'].min()), int(df['year'].max())
            year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        
        with col3:
            selected_sex = st.multiselect(
                "Select Sex",
                options=sorted(df['sex'].unique()),
                default=sorted(df['sex'].unique())
            )
        
        filtered_df = df[
            (df['country'].isin(selected_countries)) &
            (df['year'] >= year_range[0]) &
            (df['year'] <= year_range[1]) &
            (df['sex'].isin(selected_sex))
        ]
        
        st.markdown("<h2 class='sub-header'>Filtered Data</h2>", unsafe_allow_html=True)
        st.dataframe(filtered_df)
        
        st.markdown("<h2 class='sub-header'>Visualizations</h2>", unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Select Visualization",
            ["Mortality Trends by Country", "Mortality Distribution", "Age-Specific Mortality", "Heatmap by Country and Year"]
        )
        
        if viz_type == "Mortality Trends by Country":
            country_year_data = filtered_df.groupby(['country', 'year'])['total'].mean().reset_index()
            
            fig = px.line(
                country_year_data,
                x='year',
                y='total',
                color='country',
                title='Mortality Trends by Country',
                labels={'total': 'Average Mortality', 'year': 'Year'},
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Mortality Distribution":
            fig = px.box(
                filtered_df,
                x='country',
                y='total',
                color='country',
                title='Mortality Distribution by Country',
                labels={'total': 'Mortality', 'country': 'Country'},
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Age-Specific Mortality":
            age_cols = [col for col in df.columns if col.startswith('d')]
            
            if age_cols:
                selected_country = st.selectbox(
                    "Select Country for Age Analysis",
                    options=selected_countries
                )
                
                selected_year = st.slider(
                    "Select Year",
                    min_value=year_range[0],
                    max_value=year_range[1],
                    value=year_range[1]
                )

                age_data = filtered_df[
                    (filtered_df['country'] == selected_country) &
                    (filtered_df['year'] == selected_year)
                ]
                
                if not age_data.empty:
                    age_data_melted = pd.melt(
                        age_data,
                        id_vars=['country', 'year', 'sex'],
                        value_vars=age_cols,
                        var_name='age_group',
                        value_name='mortality'
                    )
                    
                    fig = px.bar(
                        age_data_melted,
                        x='age_group',
                        y='mortality',
                        color='sex',
                        barmode='group',
                        title=f'Age-Specific Mortality for {selected_country} in {selected_year}',
                        labels={'mortality': 'Mortality', 'age_group': 'Age Group'},
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {selected_country} in {selected_year}")
            else:
                st.warning("Age-specific data not available in the dataset")
        
        elif viz_type == "Heatmap by Country and Year":
            heatmap_data = filtered_df.pivot_table(
                index='country',
                columns='year',
                values='total',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Year", y="Country", color="Mortality"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='YlOrRd',
                title='Mortality Heatmap by Country and Year'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h2 class='sub-header'>Statistical Summary</h2>", unsafe_allow_html=True)
        
        summary_stats = filtered_df.groupby('country')['total'].describe()
        
        st.dataframe(summary_stats)
        
        st.download_button(
            label="Download Filtered Data",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_mortality_data.csv",
            mime="text/csv"
        )
    
    elif page == "Anomaly Detection":
        st.markdown("<h1 class='main-header'>Anomaly Detection</h1>", unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Detection Parameters</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            detection_method = st.selectbox(
                "Detection Method",
                options=["zscore", "iqr", "moving_avg"],
                format_func=lambda x: {
                    "zscore": "Z-Score",
                    "iqr": "IQR (Interquartile Range)",
                    "moving_avg": "Moving Average"
                }[x]
            )
        
        with col2:
            threshold = st.slider(
                "Threshold",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Higher threshold = fewer anomalies detected"
            )
        
        with col3:
            selected_countries = st.multiselect(
                "Select Countries",
                options=sorted(df['country'].unique()),
                default=sorted(df['country'].unique())[:3]
            )
        
        filtered_df = df[df['country'].isin(selected_countries)]
        
        anomaly_df = detect_anomalies(filtered_df, method=detection_method, threshold=threshold)
        
        anomaly_df = match_with_historical_events(anomaly_df)

        st.markdown("<h2 class='sub-header'>Anomaly Statistics</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_anomalies = anomaly_df['is_anomaly'].sum()
            st.metric("Total Anomalies Detected", total_anomalies)
        
        with col2:
            anomaly_percent = (total_anomalies / len(anomaly_df)) * 100
            st.metric("Percentage of Data Points", f"{anomaly_percent:.2f}%")
        
        with col3:
            matched_events = anomaly_df[anomaly_df['has_matched_event'] == True]['is_anomaly'].sum()
            if total_anomalies > 0:
                matched_percent = (matched_events / total_anomalies) * 100
            else:
                matched_percent = 0
            st.metric("Matched with Historical Events", f"{matched_percent:.2f}%")
        
        st.markdown("<h2 class='sub-header'>Detected Anomalies</h2>", unsafe_allow_html=True)
        
        anomalies_only = anomaly_df[anomaly_df['is_anomaly'] == 1].sort_values(['country', 'year'])
        
        st.dataframe(anomalies_only[['country', 'year', 'sex', 'total', 'anomaly_score', 'matched_events', 'event_type']])
        
        st.markdown("<h2 class='sub-header'>Visualizations</h2>", unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Select Visualization",
            ["Anomaly Timeline", "Anomaly Distribution by Country", "Anomaly Scores", "Event Type Distribution"]
        )

        if viz_type == "Anomaly Timeline":
            anomaly_timeline = anomalies_only.groupby('year')['is_anomaly'].sum().reset_index()
            
            fig = px.line(
                anomaly_timeline,
                x='year',
                y='is_anomaly',
                markers=True,
                title='Anomaly Timeline',
                labels={'is_anomaly': 'Number of Anomalies', 'year': 'Year'},
                template='plotly_white'
            )
            
            historical_events = get_historical_events()
            for year, events in historical_events.items():
                if year in anomaly_timeline['year'].values:
                    fig.add_annotation(
                        x=year,
                        y=anomaly_timeline[anomaly_timeline['year'] == year]['is_anomaly'].values[0],
                        text=events[0],
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40
                    )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Anomaly Distribution by Country":
            country_anomalies = anomalies_only.groupby('country')['is_anomaly'].sum().reset_index()
            country_anomalies = country_anomalies.sort_values('is_anomaly', ascending=False)

            fig = px.bar(
                country_anomalies,
                x='country',
                y='is_anomaly',
                color='country',
                title='Anomaly Distribution by Country',
                labels={'is_anomaly': 'Number of Anomalies', 'country': 'Country'},
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Anomaly Scores":
            fig = px.scatter(
                anomaly_df,
                x='year',
                y='anomaly_score',
                color='country',
                size=abs(anomaly_df['anomaly_score']) + 1, 
                hover_data=['total'],
                title='Anomaly Scores by Year',
                labels={'anomaly_score': 'Anomaly Score', 'year': 'Year'},
                template='plotly_white'
            )
            
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Upper Threshold ({threshold})")
            fig.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text=f"Lower Threshold (-{threshold})")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Event Type Distribution":
            event_counts = anomalies_only['event_type'].value_counts().reset_index()
            event_counts.columns = ['event_type', 'count']

            fig = px.pie(
                event_counts,
                values='count',
                names='event_type',
                title='Distribution of Anomalies by Event Type',
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h2 class='sub-header'>Country-Specific Analysis</h2>", unsafe_allow_html=True)
        
        selected_country = st.selectbox(
            "Select Country for Detailed Analysis",
            options=selected_countries
        )
        
        country_data = anomaly_df[anomaly_df['country'] == selected_country].sort_values('year')
        country_anomalies = country_data[country_data['is_anomaly'] == 1]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['total'],
                mode='lines+markers',
                name='Mortality',
                line=dict(color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=country_data['year'],
                y=country_data['anomaly_score'],
                mode='lines',
                name='Anomaly Score',
                line=dict(color='orange', dash='dot')
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=country_anomalies['year'],
                y=country_anomalies['total'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Anomalies'
            )
        )
        
        fig.update_layout(
            title=f'Mortality and Anomaly Analysis for {selected_country}',
            xaxis_title='Year',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Mortality", secondary_y=False)
        fig.update_yaxes(title_text="Anomaly Score", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Anomalies for {selected_country}")
        st.dataframe(country_anomalies[['year', 'sex', 'total', 'anomaly_score', 'matched_events', 'event_type']])
        
        st.download_button(
            label="Download Anomaly Data",
            data=anomalies_only.to_csv(index=False),
            file_name="mortality_anomalies.csv",
            mime="text/csv"
        )
    
    elif page == "Historical Correlation":
        st.markdown("<h1 class='main-header'>Historical Correlation Analysis</h1>", unsafe_allow_html=True)
        
        historical_events = get_historical_events()
        
        selected_countries = st.multiselect(
            "Select Countries",
            options=sorted(df['country'].unique()),
            default=sorted(df['country'].unique())[:3]
        )
        
        filtered_df = df[df['country'].isin(selected_countries)]
        
        detection_method = st.selectbox(
            "Detection Method",
            options=["zscore", "iqr", "moving_avg"],
            format_func=lambda x: {
                "zscore": "Z-Score",
                "iqr": "IQR (Interquartile Range)",
                "moving_avg": "Moving Average"
            }[x]
        )
        
        threshold = st.slider(
            "Threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Higher threshold = fewer anomalies detected"
        )
        
        anomaly_df = detect_anomalies(filtered_df, method=detection_method, threshold=threshold)
        
        anomaly_df = match_with_historical_events(anomaly_df)
        
        st.markdown("<h2 class='sub-header'>Historical Events Timeline</h2>", unsafe_allow_html=True)
        
        events_data = []
        for year, events in historical_events.items():
            if year >= df['year'].min() and year <= df['year'].max():
                for event in events:
                    event_type = "Other"
                    for type_name, years in {
                        'Pandemic': {1968, 1969, 1970, 1981, 1982, 1983, 2002, 2003, 2009, 2010, 2014, 2015, 2018, 2019, 2020, 2021},
                        'Conflict': {1991, 1992, 1993, 1994, 1995, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2011, 2012, 2013, 2014, 2015},
                        'Natural Disaster': {1970, 1976, 1985, 1995, 2004, 2005, 2008, 2010, 2011},
                        'Policy Change': {1990, 1991, 2010},
                        'Heat Wave': {2003, 2010, 2015}
                    }.items():
                        if year in years:
                            event_type = type_name
                            break
                    
                    events_data.append({
                        'year': year,
                        'event': event,
                        'event_type': event_type
                    })
        
        events_df = pd.DataFrame(events_data)
        
        fig = px.scatter(
            events_df,
            x='year',
            y='event_type',
            color='event_type',
            hover_data=['event'],
            size=[10] * len(events_df),
            title='Timeline of Historical Events',
            labels={'year': 'Year', 'event_type': 'Event Type'},
            template='plotly_white'
        )
        
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(yaxis=dict(categoryorder='category ascending'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
        
        years = sorted(df['year'].unique())
        correlation_data = []
        
        for year in years:
            anomalies_detected = int(year in anomaly_df[anomaly_df['is_anomaly'] == 1]['year'].values)
            
            known_event = int(year in historical_events)

            if year in anomaly_df[anomaly_df['is_anomaly'] == 1]['year'].values:
                anomaly_count = len(anomaly_df[(anomaly_df['year'] == year) & (anomaly_df['is_anomaly'] == 1)])
            else:
                anomaly_count = 0
            
            if year in historical_events:
                event_count = len(historical_events[year])
            else:
                event_count = 0
            
            correlation_data.append({
                'year': year,
                'anomalies_detected': anomalies_detected,
                'anomaly_count': anomaly_count,
                'known_event': known_event,
                'event_count': event_count
            })
        
        correlation_df = pd.DataFrame(correlation_data)
        
        if (correlation_df['anomalies_detected'].var() > 0 and correlation_df['known_event'].var() > 0):
            correlation = correlation_df['anomalies_detected'].corr(correlation_df['known_event'])
            st.metric("Correlation between anomaly detection and known events", f"{correlation:.4f}")
        else:
            st.warning("Cannot calculate correlation: insufficient variation in data")
        
        count_correlation = correlation_df['anomaly_count'].corr(correlation_df['event_count'])
        st.metric("Correlation between anomaly count and event count", f"{count_correlation:.4f}")

        contingency = pd.crosstab(
            correlation_df['anomalies_detected'],
            correlation_df['known_event'],
            rownames=['Anomalies Detected'],
            colnames=['Known Event Occurred']
        )
        
        try:
            true_positives = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
            false_positives = contingency.loc[1, 0] if 1 in contingency.index and 0 in contingency.columns else 0
            false_negatives = contingency.loc[0, 1] if 0 in contingency.index and 1 in contingency.columns else 0
            true_negatives = contingency.loc[0, 0] if 0 in contingency.index and 0 in contingency.columns else 0

            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            if sum(contingency.values.flatten()) > 0:
                accuracy = (true_positives + true_negatives) / sum(contingency.values.flatten())
            else:
                accuracy = 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", f"{precision:.4f}")
                st.caption("Proportion of detected anomalies that correspond to known events")
            
            with col2:
                st.metric("Recall", f"{recall:.4f}")
                st.caption("Proportion of known events that were detected as anomalies")
            
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.caption("Harmonic mean of precision and recall")
            
            with col4:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.caption("Overall accuracy of event detection")
            
            st.subheader("Contingency Table")
            st.dataframe(contingency)
            
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
        
        st.markdown("<h2 class='sub-header'>Precision-Recall Visualization</h2>", unsafe_allow_html=True)
        
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[recall],
                y=[precision],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name=f'Current Model (F1: {f1:.2f})'
            ))
            
            f1_x = np.linspace(0.01, 1, 100)
            f1_y = f1 * f1_x / (2 * f1_x - f1) if f1 > 0 else np.zeros_like(f1_x)
            
            fig.add_trace(go.Scatter(
                x=f1_x,
                y=f1_y,
                mode='lines',
                line=dict(color='green', dash='dash'),
                name=f'F1={f1:.2f}'
            ))
            
            fig.update_layout(
                title='Precision-Recall for Historical Event Detection',
                xaxis_title='Recall',
                yaxis_title='Precision',
                xaxis=dict(range=[0, 1.05]),
                yaxis=dict(range=[0, 1.05]),
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating precision-recall visualization: {e}")
        
        st.markdown("<h2 class='sub-header'>Anomalies by Event Type</h2>", unsafe_allow_html=True)
        
        event_type_counts = anomaly_df[anomaly_df['is_anomaly'] == 1]['event_type'].value_counts().reset_index()
        event_type_counts.columns = ['event_type', 'count']
        
        event_type_counts['percentage'] = event_type_counts['count'] / event_type_counts['count'].sum() * 100
        
        st.dataframe(event_type_counts)
        
        fig = px.pie(
            event_type_counts,
            values='count',
            names='event_type',
            title='Distribution of Anomalies by Event Type',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h2 class='sub-header'>Timeline by Event Type</h2>", unsafe_allow_html=True)
        
        event_by_year = pd.crosstab(
            anomaly_df[anomaly_df['is_anomaly'] == 1]['year'],
            anomaly_df[anomaly_df['is_anomaly'] == 1]['event_type']
        )
        
        event_by_year = event_by_year.fillna(0)
        
        fig = px.bar(
            event_by_year,
            x=event_by_year.index,
            y=event_by_year.columns,
            title='Mortality Anomalies by Year and Event Type',
            labels={'x': 'Year', 'value': 'Number of Anomalies', 'variable': 'Event Type'},
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Excess Mortality":
        st.markdown("<h1 class='main-header'>Excess Mortality Analysis</h1>", unsafe_allow_html=True)

        selected_countries = st.multiselect(
            "Select Countries",
            options=sorted(df['country'].unique()),
            default=sorted(df['country'].unique())[:3]
        )
        
        filtered_df = df[df['country'].isin(selected_countries)]
        
        detection_method = st.selectbox(
            "Detection Method",
            options=["zscore", "iqr", "moving_avg"],
            format_func=lambda x: {
                "zscore": "Z-Score",
                "iqr": "IQR (Interquartile Range)",
                "moving_avg": "Moving Average"
            }[x]
        )
        
        threshold = st.slider(
            "Threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Higher threshold = fewer anomalies detected"
        )
        
        window_size = st.slider(
            "Window Size for Expected Mortality",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Number of previous years to use for calculating expected mortality"
        )
        
        anomaly_df = detect_anomalies(filtered_df, method=detection_method, threshold=threshold)
        
        anomaly_df = match_with_historical_events(anomaly_df)
        
        excess_df = calculate_excess_mortality(anomaly_df, window_size=window_size)

        st.markdown("<h2 class='sub-header'>Excess Mortality Statistics</h2>", unsafe_allow_html=True)

        excess_anomalies = excess_df[
            (excess_df['is_anomaly'] == 1) & 
            (excess_df['expected_mortality'].notna())
        ].sort_values(['country', 'year'])
        
        if not excess_anomalies.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_excess = excess_anomalies['excess_mortality'].mean()
                st.metric("Average Excess Mortality", f"{avg_excess:.2f}")
            
            with col2:
                avg_excess_pct = excess_anomalies['excess_percent'].mean()
                st.metric("Average Excess Percentage", f"{avg_excess_pct:.2f}%")
            
            with col3:
                significant_count = excess_anomalies['significant'].sum()
                significant_pct = (significant_count / len(excess_anomalies)) * 100
                st.metric("Significant Anomalies", f"{significant_count} ({significant_pct:.2f}%)")
            
            st.dataframe(excess_anomalies[[
                'country', 'year', 'total', 'expected_mortality', 
                'excess_mortality', 'excess_percent', 'significant',
                'event_type', 'matched_events'
            ]])

            if 'event_type' in excess_anomalies.columns:
                event_summary = excess_anomalies.groupby('event_type').agg(
                    count=('country', 'count'),
                    avg_excess=('excess_mortality', 'mean'),
                    avg_excess_percent=('excess_percent', 'mean'),
                    significant_count=('significant', 'sum'),
                    significant_percent=('significant', lambda x: x.mean() * 100)
                ).reset_index()
                
                st.markdown("<h2 class='sub-header'>Excess Mortality by Event Type</h2>", unsafe_allow_html=True)
                st.dataframe(event_summary)

            st.markdown("<h2 class='sub-header'>Visualizations</h2>", unsafe_allow_html=True)
            
            viz_type = st.selectbox(
                "Select Visualization",
                ["Excess Mortality by Country", "Excess Mortality by Year", "Excess Mortality Distribution", "Top Anomalies Comparison", "Excess Mortality by Event Type"]
            )
            
            if viz_type == "Excess Mortality by Country":
                country_excess = excess_anomalies.groupby('country').agg(
                    avg_excess_percent=('excess_percent', 'mean'),
                    max_excess_percent=('excess_percent', 'max'),
                    count=('year', 'count')
                ).reset_index()
                
                country_excess = country_excess.sort_values('avg_excess_percent', ascending=False)
                
                fig = px.bar(
                    country_excess,
                    x='country',
                    y='avg_excess_percent',
                    color='country',
                    title='Average Excess Mortality by Country',
                    labels={'avg_excess_percent': 'Average Excess Mortality (%)', 'country': 'Country'},
                    template='plotly_white'
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Excess Mortality by Year":
                year_excess = excess_anomalies.groupby('year').agg(
                    avg_excess_percent=('excess_percent', 'mean'),
                    max_excess_percent=('excess_percent', 'max'),
                    count=('country', 'count')
                ).reset_index()
                
                year_excess = year_excess.sort_values('year')

                fig = px.bar(
                    year_excess,
                    x='year',
                    y='avg_excess_percent',
                    title='Average Excess Mortality by Year',
                    labels={'avg_excess_percent': 'Average Excess Mortality (%)', 'year': 'Year'},
                    template='plotly_white'
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Excess Mortality Distribution":
                fig = px.histogram(
                    excess_anomalies,
                    x='excess_percent',
                    nbins=20,
                    title='Distribution of Excess Mortality Percentage',
                    labels={'excess_percent': 'Excess Mortality (%)'},
                    template='plotly_white'
                )
                
                fig.add_vline(x=0, line_dash="dash", line_color="red")

                kde_x = np.linspace(
                    excess_anomalies['excess_percent'].min(),
                    excess_anomalies['excess_percent'].max(),
                    100
                )
                kde = stats.gaussian_kde(excess_anomalies['excess_percent'].dropna())
                kde_y = kde(kde_x)
                
                hist_max = np.histogram(excess_anomalies['excess_percent'], bins=20)[0].max()
                kde_max = kde_y.max()
                scale_factor = hist_max / kde_max
                
                fig.add_scatter(
                    x=kde_x,
                    y=kde_y * scale_factor,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Density'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Top Anomalies Comparison":
                top_excess = excess_anomalies.nlargest(10, 'excess_percent')
                
                fig = make_subplots(
                    rows=5, cols=2,
                    subplot_titles=[f"{row['country']} - {row['year']} (+{row['excess_percent']:.1f}%)" 
                                   for _, row in top_excess.iterrows()]
                )
                
                for i, (_, row) in enumerate(top_excess.iterrows()):
                    r, c = i // 2 + 1, i % 2 + 1

                    fig.add_trace(
                        go.Bar(
                            x=['Expected'],
                            y=[row['expected_mortality']],
                            name='Expected',
                            marker_color='blue',
                            showlegend=i==0
                        ),
                        row=r, col=c
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=['Actual'],
                            y=[row['total']],
                            name='Actual',
                            marker_color='red',
                            showlegend=i==0
                        ),
                        row=r, col=c
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=['Expected'],
                            y=[row['expected_mortality']],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=[row['upper_bound'] - row['expected_mortality']],
                                arrayminus=[row['expected_mortality'] - row['lower_bound']]
                            ),
                            mode='markers',
                            marker=dict(color='rgba(0,0,0,0)'),
                            showlegend=False
                        ),
                        row=r, col=c
                    )

                fig.update_layout(
                    height=1000,
                    title_text='Top 10 Mortality Anomalies: Expected vs. Actual',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Excess Mortality by Event Type":
                if 'event_type' in excess_anomalies.columns:
                    fig = px.box(
                        excess_anomalies,
                        x='event_type',
                        y='excess_percent',
                        color='event_type',
                        title='Excess Mortality by Event Type',
                        labels={'excess_percent': 'Excess Mortality (count)', 'event_type': 'Event Type'},
                        template='plotly_white'
                    )
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Event type information not available")
            
            st.markdown("<h2 class='sub-header'>Country-Specific Analysis</h2>", unsafe_allow_html=True)
            
            selected_country = st.selectbox(
                "Select Country for Detailed Analysis",
                options=excess_anomalies['country'].unique()
            )
            
            country_excess = excess_anomalies[excess_anomalies['country'] == selected_country].sort_values('year')
            
            if not country_excess.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(
                        x=country_excess['year'],
                        y=country_excess['total'],
                        mode='lines+markers',
                        name='Actual Mortality',
                        line=dict(color='red')
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=country_excess['year'],
                        y=country_excess['expected_mortality'],
                        mode='lines+markers',
                        name='Expected Mortality',
                        line=dict(color='blue')
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=country_excess['year'],
                        y=country_excess['excess_percent'],
                        mode='lines',
                        name='Excess Percentage',
                        line=dict(color='green', dash='dot')
                    ),
                    secondary_y=True
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=country_excess['year'],
                        y=country_excess['upper_bound'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=country_excess['year'],
                        y=country_excess['lower_bound'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.1)',
                        name='95% Confidence Interval'
                    )
                )
                
                fig.update_layout(
                    title=f'Excess Mortality Analysis for {selected_country}',
                    xaxis_title='Year',
                    template='plotly_white'
                )
                
                fig.update_yaxes(title_text="Mortality", secondary_y=False)
                fig.update_yaxes(title_text="Excess Percentage (%)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader(f"Excess Mortality for {selected_country}")
                st.dataframe(country_excess[[
                    'year', 'total', 'expected_mortality', 
                    'excess_mortality', 'excess_percent', 'significant',
                    'event_type', 'matched_events'
                ]])
            else:
                st.warning(f"No excess mortality data available for {selected_country}")
        else:
            st.warning("No anomalies with excess mortality calculations found. Try adjusting the parameters.")
        
        if not excess_anomalies.empty:
            st.download_button(
                label="Download Excess Mortality Data",
                data=excess_anomalies.to_csv(index=False),
                file_name="excess_mortality_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
