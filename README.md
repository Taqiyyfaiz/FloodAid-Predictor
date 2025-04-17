# FloodAid Predictor

A full-stack AI application built for "AI for Good: Humanitarian Aid Optimization" that uses machine learning to predict flood risk zones and optimize aid delivery routes during flood events.

## Overview

FloodAid Predictor combines geospatial analysis, machine learning, and route optimization to help disaster response teams identify high-risk flood areas and determine the most efficient routes for delivering aid. The system generates synthetic flood risk data based on elevation, flow accumulation, and rainfall intensity, then applies optimization algorithms to plan the safest and most efficient routes to affected areas.

## Key Features

- **Flood Risk Prediction**: Uses topographic data and rainfall information to predict areas at high risk of flooding
- **Route Optimization**: Calculates optimal routes for aid delivery, avoiding high-risk flood zones
- **Interactive Visualization**: Provides real-time visualizations of flood risk areas and optimized delivery routes
- **Multi-region Support**: Includes data for multiple flood-prone regions (Mumbai, Chennai, Dhaka, Jakarta)
- **Scenario Simulation**: Allows users to adjust rainfall intensity to simulate different flooding scenarios

## Tech Stack

- **Frontend**: 
  - [Streamlit](https://streamlit.io/) for interactive web interface and data visualization
  - [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium) for embedding Folium maps in Streamlit

- **Data Processing**: 
  - [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation and numerical operations
  - [GeoPandas](https://geopandas.org/) for geospatial data processing
  - [SQLite3](https://www.sqlite.org/) for local database storage

- **Geospatial Analysis**:
  - [Shapely](https://shapely.readthedocs.io/) for geometric operations
  - [PyProj](https://pyproj4.github.io/pyproj/stable/) for coordinate system transformations
  - [Folium](https://python-visualization.github.io/folium/) for interactive maps visualization

- **Machine Learning**: 
  - [Scikit-learn](https://scikit-learn.org/) for flood prediction models
  - [Joblib](https://joblib.readthedocs.io/) for model serialization

- **Optimization**: 
  - [Google OR-Tools](https://developers.google.com/optimization) for route optimization algorithms

- **Visualization**: 
  - [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data plotting
  - [Folium](https://python-visualization.github.io/folium/) for interactive maps

- **API Integration**:
  - [Requests](https://requests.readthedocs.io/) for external API calls
  - [python-dotenv](https://github.com/theskumar/python-dotenv) for environment variable management

- **Reporting & Export**:
  - [XlsxWriter](https://xlsxwriter.readthedocs.io/) for Excel report generation
  - [Pillow](https://python-pillow.org/) for image processing

## Project Structure

```
FloodAid-Predictor/
├── app.py                 # Main Streamlit application
├── config/                # Configuration files and parameters
├── components/            # UI components for the dashboard
│   ├── map_component.py   # Interactive map visualization
│   ├── dashboard_component.py # Main dashboard UI
│   └── analytics_component.py # Analytics and reporting
├── data/                  # Data storage
├── models/                # ML models for prediction and optimization
│   ├── flood_prediction.py         # Flood risk prediction model
│   └── route_optimization.py       # Route optimization algorithms
├── utils/                 # Utility functions
│   ├── geo_utils.py       # Geospatial calculations and mapping
│   ├── weather_api.py     # Weather data retrieval
│   ├── csv_data_manager.py # CSV data management
│   └── database.py        # Database interactions
└── requirements.txt       # Dependencies
```

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FloodAid-Predictor.git
   cd FloodAid-Predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - For demonstration purposes, you can leave `OPENWEATHER_API_KEY=DEMO_MODE` to use synthetic weather data
   - For real weather data, replace with your actual OpenWeather API key:
     ```
     OPENWEATHER_API_KEY=your_actual_api_key_here
     ```
   - You can get a free API key by registering at [OpenWeatherMap](https://openweathermap.org/api)

## Running the Application

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

## Detailed Library Requirements

The application requires the following Python libraries:

```
numpy==1.24.3
pandas==2.0.1
scikit-learn==1.3.0
ortools==9.6.2534
geopandas==0.13.2
folium==0.14.0
streamlit==1.22.0
matplotlib==3.7.1
seaborn==0.12.2
requests==2.29.0
python-dotenv==1.0.0
sqlite3-api==0.1.0
pyproj==3.5.0
shapely==2.0.1
joblib==1.2.0
xlsxwriter==3.1.2
pillow==10.0.0
streamlit-folium==0.13.0
```

## Using Synthetic Data vs Real Data

By default, the application runs in demonstration mode using synthetic weather data. This allows you to try out all features without needing any API keys.

To use real weather data:
1. Register for a free API key at [OpenWeatherMap](https://openweathermap.org/api)
2. Add your API key to the `.env` file: `OPENWEATHER_API_KEY=your_actual_api_key_here`
3. Restart the application

The application will automatically detect your API key and use real weather data instead of synthetic data.

## Key Functionality

### Flood Risk Prediction
The application uses a combination of elevation data, flow accumulation calculations, and rainfall information to predict areas at high risk of flooding. The prediction model considers:

- Terrain elevation
- Water flow patterns based on topography
- Current and forecasted rainfall intensity
- Historical flood data (when available)

### Route Optimization
For aid delivery planning, the application uses optimization algorithms to:

- Identify the shortest paths between aid centers and affected areas
- Avoid high-risk flood zones in route planning
- Prioritize areas based on population and severity of impact
- Calculate estimated travel times considering route conditions

### Visualization Features
The interactive maps display:
- Color-coded flood risk zones (low, medium, high, severe)
- Village locations with risk indicators (red for high-risk, green for low-risk)
- Aid center locations (green markers with hospital icon)
- Historical flood locations (blue circles)
- Optimized routes between aid centers and affected areas (blue lines)

## Future Improvements

### Data Sources and Integration
- Integration with real-time flood monitoring systems
- Connection to actual weather APIs (OpenWeather, DarkSky)
- Incorporation of satellite imagery for flood extent mapping
- Integration with ground-level sensor data

### Machine Learning Enhancements
- Deep learning models for improved prediction accuracy
- Time-series forecasting for flood progression modeling
- Computer vision for flood image analysis from satellites or drones
- Ensemble methods combining multiple prediction approaches

### Route Optimization
- Multi-objective optimization balancing time, risk, and coverage
- Dynamic rerouting based on changing flood conditions
- Integration with actual road network data (OSM)
- Support for different vehicle types and capabilities

### UI/UX Improvements
- Mobile-responsive design for field workers
- Offline capability for areas with limited connectivity
- Customizable dashboards for different user roles
- Alert and notification system for rapidly changing conditions

### Deployment and Scaling
- Containerization with Docker for easy deployment
- Cloud infrastructure for increased computational capacity
- API endpoints for integration with other emergency systems
- Automated data pipeline for continuous updates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project was developed as part of the "AI for Good: Humanitarian Aid Optimization" initiative
- Special thanks to all contributors and the open-source community

---

> **Note:** FloodAid Predictor is an ongoing project with significant potential for improvement and expansion. We are actively seeking collaborators with expertise in machine learning, geospatial analysis, disaster management, and full-stack development to help enhance this tool. If you're passionate about using technology for humanitarian purposes, please consider contributing. Every contribution, whether it's code, documentation, or ideas, helps make disaster response more effective and potentially saves lives. Feel free to open issues, submit pull requests, or contact the maintainers directly.