from flask import Flask, jsonify, request, render_template
from src.weather import VA, Namibia, Chile, SRO, awoa, Wolongbar, Brazil
from src.skyroof import latest, read_log, read_action_log, LOG_PATH, ACTION_LOG_PATH
from src.storage import save_reading, get_history_for_chart
from src.chart import get_forecast
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
    log_path    = request.args.get('log',    LOG_PATH)
    action_path = request.args.get('actions', ACTION_LOG_PATH)
    entries = read_log(log_path)
    if not entries:
        return jsonify({'error': f'No data found at {log_path}'}), 404
    save_reading(entries[0])
    return jsonify({
        'latest':  entries[0],
        'history': entries,
        'actions': read_action_log(action_path),
    })

@app.route('/api/chart/<site>')
def chart_data(site):
    result = get_forecast(site)
    if not result:
        return jsonify({'error': f'Forecast unavailable for {site}'}), 503
    result['history'] = get_history_for_chart()
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)