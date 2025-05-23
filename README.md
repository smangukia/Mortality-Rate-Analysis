# Mortality Rate Analysis

A comprehensive data science project featuring advanced machine learning algorithms and interactive visualizations for analyzing global mortality patterns, detecting anomalies, and correlating them with historical events. This repository contains both the original Jupyter notebook with detailed data analysis and an interactive Streamlit web application.

## 🎯 Project Overview

This project combines statistical analysis, machine learning, and time series forecasting to identify unusual mortality patterns across different countries and time periods. The system implements multiple anomaly detection algorithms and provides an interactive Streamlit dashboard for real-time analysis.

## 📊 Screenshots

### Streamlit Dashboard - Home Page
![Home Page](screenshots/HomePage.png)

### Data Exploration
![Data Exploration](screenshots/DataExploration1.png)
![Data Exploration](screenshots/DataExploration2.png)

### Anomaly Detection
![Anomaly Detection](screenshots/AnomalyDetection.png)

### Historical Correlation
![Historical Correlation](screenshots/HistoricalCorrelation.png)

### Excess Mortality
![Excess Mortality](screenshots/ExcessMortality.png)

## 🌟 Features

### Jupyter Notebook Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis of mortality data
- **Data Preprocessing**: Data cleaning, transformation, and preparation
- **Statistical Modeling**: Implementation of anomaly detection algorithms
- **Machine Learning Models**: Isolation Forest, One-Class SVM, LSTM Autoencoders
- **Time Series Analysis**: SARIMA models and LSTM neural networks
- **Visualization**: Detailed plots and charts for data insights
- **Research Documentation**: Step-by-step analysis with explanations

### Interactive Streamlit Dashboard
- **Data Exploration**: Analyze mortality data by country, year, and demographic factors
- **Anomaly Detection**: Identify unusual mortality patterns using multiple methods:
  - Z-Score method
  - IQR (Interquartile Range) method
  - Moving Average method
  - **Ensemble Approach**: Combining statistical and ML methods achieving **81.25% F1-score**

- **Historical Correlation**: Correlate mortality anomalies with historical events:
  - Pandemics
  - Conflicts
  - Natural disasters
  - Policy changes
  - Heat waves

- **Excess Mortality**: Calculate and visualize excess mortality during anomalous periods
- **Interactive Visualizations**: Dynamic charts and plots with user controls
- **Data Export**: Download filtered data and analysis results


## 🔧 Installation

1. Clone this repository:

```shellscript
git clone https://github.com/smangukia/Mortality-Rate-Analysis.git
cd Mortality-Rate-Analysis
```


2. Create a virtual environment (optional but recommended):

```shellscript
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


3. Install the required packages:

```shellscript
pip install -r requirements.txt
```


## 🚀 Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:

```shellscript
jupyter notebook
```


2. Open `mortality_analysis.ipynb` in your browser
3. Run the cells sequentially to reproduce the analysis


### Running the Streamlit Dashboard

1. Run the Streamlit app:

```shellscript
streamlit run app.py
```

2. Open your browser and navigate to the URL displayed in the terminal (typically [http://localhost:8501](http://localhost:8501))
3. Navigate through the different pages using the sidebar:

1. Home
2. Data Exploration
3. Anomaly Detection
4. Historical Correlation
5. Excess Mortality


4. Adjust parameters using the interactive controls and explore visualizations


## 📊 Data

### Data Source

The project uses data from the **Human Mortality Database (HMD)**, a collaborative project between the University of California, Berkeley (USA) and the Max Planck Institute for Demographic Research (Germany). The HMD provides detailed mortality and population data for over 40 countries or areas, with some series extending back to the 19th century.

- **Website**: [Human Mortality Database](https://www.mortality.org/)
- **Coverage**: 40+ countries/regions
- **Time Span**: Some data extends back to the 1800s
- **Quality**: High-quality, validated demographic data
- **Standardization**: Consistent methods applied across all countries

The HMD is widely used in demographic research, epidemiology, and public health studies due to its comprehensive coverage and methodological consistency.

### Data Structure

The application uses mortality data with the following structure:
- **Country**: Country identifier
- **Year**: Year of data collection
- **Sex**: Gender classification (1=Male, 2=Female)
- **Total**: Total mortality count
- **Age-specific columns**: Mortality by age groups (d0, d1, d5, d10, etc.)

*Note: If the original data file is not found, the application will generate sample data for demonstration purposes.*


## 🔬 Methodology

### Anomaly Detection Methods

1. **Z-Score Method**
   - Identifies data points that deviate significantly from the mean
   - Configurable threshold (typically 2-3 standard deviations)

2. **IQR (Interquartile Range)**
   - Identifies outliers based on the interquartile range
   - Robust to extreme values

3. **Moving Average**
   - Identifies points that deviate from local trends
   - Adaptive to temporal patterns

4. **Machine Learning Approaches**
   - Isolation Forest for unsupervised anomaly detection
   - One-Class SVM for novelty detection
   - LSTM Autoencoders for temporal pattern recognition

### Excess Mortality Calculation

1. **Baseline Estimation**: Calculate expected mortality using historical averages
2. **Confidence Intervals**: Determine 95% confidence bounds using t-distribution
3. **Excess Calculation**: Compute difference between actual and expected mortality
4. **Statistical Significance**: Identify statistically significant deviations

### Historical Event Correlation

The system includes a comprehensive database of historical events:
- **Pandemics**: COVID-19, H1N1, SARS, HIV/AIDS, Hong Kong Flu
- **Conflicts**: Wars, civil conflicts, terrorist attacks
- **Natural Disasters**: Earthquakes, tsunamis, hurricanes, cyclones
- **Policy Changes**: Healthcare reforms, reunifications
- **Environmental Events**: Heat waves, climate-related mortality

## 🖥️ Dashboard Pages

### 1. Home
- Project overview and key statistics
- Global mortality trends
- Dataset summary and features

### 2. Data Exploration
- Interactive filtering by country, year, and demographics
- Multiple visualization types (trends, distributions, heatmaps)
- Age-specific mortality analysis
- Statistical summaries and data export

### 3. Anomaly Detection
- Real-time anomaly detection with adjustable parameters
- Multiple detection methods comparison
- Anomaly timeline and distribution analysis
- Country-specific detailed analysis

### 4. Historical Correlation
- Timeline of historical events
- Correlation analysis between anomalies and events
- Precision-recall metrics and performance evaluation
- Event type classification and analysis

### 5. Excess Mortality
- Excess mortality calculation and visualization
- Statistical significance testing
- Comparison of expected vs. actual mortality
- Event-specific excess mortality analysis

## 🔧 Customization Options

- **Historical Events**: Add or modify the historical events database
- **Detection Parameters**: Adjust anomaly detection thresholds and methods
- **Visualization Styles**: Customize color schemes and chart types
- **Data Sources**: Integrate additional mortality datasets
- **Machine Learning Models**: Implement additional anomaly detection algorithms


## 🔮 Future Improvements

- Add geographic visualizations with maps
- Implement machine learning models for mortality prediction
- Add demographic breakdowns for more detailed analysis
- Implement report generation functionality
- Add data upload functionality for custom datasets
- Integrate real-time data sources
- Add more sophisticated time series analysis methods


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📈 Project Workflow

1. **Data Analysis** (`mortality_analysis.ipynb`): Start here to understand the data and methodology
2. **Web Application** (`app.py`): Interactive dashboard for exploring results and conducting further analysis
3. **Deployment**: The Streamlit app can be deployed on various platforms (Streamlit Cloud, Heroku, etc.)

## 📚 References

- Human Mortality Database. University of California, Berkeley (USA), and Max Planck Institute for Demographic Research (Germany). Available at www.mortality.org
- World Health Organization (WHO) mortality statistics
- Historical event databases and timelines
- Statistical anomaly detection literature
- Time series analysis and forecasting methods


**Note**: This project demonstrates the complete data science workflow from research and exploratory analysis in Jupyter notebooks to developing an interactive web application that can be run locally or deployed to cloud platforms.
