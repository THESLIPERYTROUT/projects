from flask import Flask, jsonify, request, render_template
from src.weather import VA, Namibia, Chile, SRO, awoa, Wolongbar, Brazil
from src.skyroof import latest, read_log, LOG_PATH
import os

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

@app.route('/skyroof')
def skyroof_dashboard():
    return render_template('skyroof.html')

@app.route('/api/skyroof')
def skyroof_data():
    log_path = request.args.get('log', LOG_PATH)
    entries = read_log(log_path)
    if not entries:
        return jsonify({'error': f'No data found at {log_path}'}), 404
    return jsonify({'latest': entries[0], 'history': entries})

if __name__ == '__main__':
    app.run(debug=True)