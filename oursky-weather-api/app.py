from flask import Flask, jsonify, request, render_template
from src.weather import VA, Namibia, Chile, SRO, awoa, Wolongbar, Brazil
from src.skyroof import latest, read_log, read_action_log, LOG_PATH, ACTION_LOG_PATH
from src.storage import save_reading, get_history_for_chart, get_night_data
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

@app.route('/night')
@app.route('/night/<date_str>')
def night_view(date_str=None):
    return render_template('night.html', default_date=date_str or '')

@app.route('/api/night/<date_str>')
def night_data(date_str):
    from datetime import datetime
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date, use YYYY-MM-DD'}), 400

    data    = get_night_data(date_str)
    actions = read_action_log(ACTION_LOG_PATH)

    # Filter actions to this night's window
    start_s = datetime.fromtimestamp(data['window_start'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    end_s   = datetime.fromtimestamp(data['window_end']   / 1000).strftime('%Y-%m-%d %H:%M:%S')
    data['actions'] = [a for a in actions if start_s <= a['timestamp'] <= end_s]

    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)