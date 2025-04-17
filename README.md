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

- **Frontend**: Streamlit for interactive web interface and data visualization
- **Data Processing**: Pandas, NumPy, GeoPandas for data manipulation and geospatial analysis
- **Machine Learning**: Scikit-learn for flood prediction models
- **Optimization**: Google OR-Tools for route optimization algorithms
- **Visualization**: Folium for interactive maps, Matplotlib and Seaborn for data plotting
- **Geospatial Processing**: Shapely for geometric operations, PyProj for coordinate transformations

## Project Structure

```
FloodAid-Predictor/
├── app.py                 # Main Streamlit application
├── config/                # Configuration files and parameters
├── components/            # UI components for the dashboard
├── data/                  # Data storage
├── models/                # ML models for prediction and optimization
│   ├── flood_prediction.py         # Flood risk prediction model
│   └── route_optimization.py       # Route optimization algorithms
├── utils/                 # Utility functions
│   ├── geo_utils.py       # Geospatial calculations and mapping
│   ├── weather_api.py     # Weather data retrieval
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

## Using Synthetic Data vs Real Data

By default, the application runs in demonstration mode using synthetic weather data. This allows you to try out all features without needing any API keys.

To use real weather data:
1. Register for a free API key at [OpenWeatherMap](https://openweathermap.org/api)
2. Add your API key to the `.env` file: `OPENWEATHER_API_KEY=your_actual_api_key_here`
3. Restart the application

The application will automatically detect your API key and use real weather data instead of synthetic data.

## Key Libraries and Dependencies

- **streamlit**: Interactive web application framework
- **pandas** & **numpy**: Data manipulation and numerical operations
- **geopandas** & **shapely**: Geospatial data processing
- **folium**: Interactive map visualization
- **scikit-learn**: Machine learning algorithms
- **ortools**: Google's optimization tools for route planning
- **matplotlib** & **seaborn**: Data visualization
- **requests**: API calls for weather data
- **python-dotenv**: Environment variable management

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

## Features

- Flood risk prediction using weather and topographic data
- Optimized route planning for aid delivery
- Interactive dashboard for real-time monitoring
- Data visualization of flood-affected areas

## Project Structure

- `data/`: Contains raw and processed datasets
- `models/`: ML models for flood prediction
- `utils/`: Utility functions for data processing and calculations
- `components/`: Dashboard components and UI elements
- `config/`: Configuration files for APIs and model parameters
- `notebooks/`: Jupyter notebooks for data exploration and model development

## Setup and Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up environment variables in `.env` file

## Running the Application

```bash
cd FloodAid
streamlit run app.py
```

## Future Enhancements

- Integration with more data sources
- Real-time weather API integration
- Mobile app for field workers
- Extension to other disaster types

---

> **Note:** FloodAid Predictor is an ongoing project with significant potential for improvement and expansion. We are actively seeking collaborators with expertise in machine learning, geospatial analysis, disaster management, and full-stack development to help enhance this tool. If you're passionate about using technology for humanitarian purposes, please consider contributing. Every contribution, whether it's code, documentation, or ideas, helps make disaster response more effective and potentially saves lives. Feel free to open issues, submit pull requests, or contact the maintainers directly.