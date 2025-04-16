# FloodAid Predictor

A prototype built for "AI for Good: Humanitarian Aid Optimization" that uses AI to predict flood risk zones and optimize aid delivery during flood events.

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