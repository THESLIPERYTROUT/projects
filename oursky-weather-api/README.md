# Oursky Weather API

## Overview
The Oursky Weather API is a Flask-based application that provides weather data for various locations. It fetches real-time weather information and exposes it through a RESTful API.

## Project Structure
```
oursky-weather-api
├── app.py              # Entry point of the Flask application
├── src
│   ├── weather.py      # Contains the Weather class and its child classes
│   └── utils.py        # Utility functions for data processing
├── requirements.txt     # Lists the project dependencies
└── README.md            # Documentation for the project
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd oursky-weather-api
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the Flask application, execute the following command:
```
python app.py
```

The application will start on `http://127.0.0.1:5000/`.

## API Endpoints
### Get Weather Data
- **Endpoint:** `/weather/<location>`
- **Method:** `GET`
- **Description:** Fetches weather data for the specified location.
- **Parameters:**
  - `location`: The name of the location (e.g., `Virginia`, `Namibia`, `Chile`, etc.).
  
- **Response:**
  - Returns a JSON object containing weather data, including cloud cover, temperature, wind speed, and precipitation chance.

## Example Request
```
GET /weather/Virginia
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.