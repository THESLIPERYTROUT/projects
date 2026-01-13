from flask import Flask, jsonify, request
from src.weather import VA, Namibia, Chile, SRO, awoa, Wolongbar, Brazil

app = Flask(__name__)

@app.route('/weather/<site>', methods=['GET'])
def get_weather_data(site):
    site_mapping = {
        'virginia': VA,
        'namibia': Namibia,
        'chile': Chile,
        'sro': SRO,
        'awoa': awoa,
        'wolongbar': Wolongbar,
        'brazil': Brazil
    }
    
    if site.lower() not in site_mapping:
        return jsonify({'error': 'Site not found'}), 404

    weather_site = site_mapping[site.lower()]()
    weather_site.fetch_data()
    
    return jsonify(weather_site.data)

if __name__ == '__main__':
    app.run(debug=True)