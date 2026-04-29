from flask import Flask, jsonify, request, render_template
from src.weather import VA, Namibia, Chile, SRO, awoa, Wolongbar, Brazil
from src.skyroof import latest, read_log, read_action_log, LOG_PATH, ACTION_LOG_PATH
from src.storage import (save_reading, get_history_for_chart, get_night_data,
                         compute_and_store_nightly_stats, get_calendar_stats)
from src.chart import get_forecast, SITES
import os
from datetime import datetime

app = Flask(__name__)

# Observatory location — override with env vars if needed
OBS_LAT  = float(os.environ.get('OBS_LAT',  39.095472))
OBS_LON  = float(os.environ.get('OBS_LON',  -77.919111))
OBS_NAME = os.environ.get('OBS_NAME', 'OurSky Observatory')


@app.route('/weather/<site>', methods=['GET'])
def get_weather_data(site):
    site_mapping = {
        'virginia': VA, 'namibia': Namibia, 'chile': Chile,
        'sro': SRO, 'awoa': awoa, 'wolongbar': Wolongbar, 'brazil': Brazil,
    }
    if site.lower() not in site_mapping:
        return jsonify({'error': 'Site not found'}), 404
    weather_site = site_mapping[site.lower()]()
    weather_site.fetch_data()
    return jsonify(weather_site.data)


@app.route('/skyroof')
def skyroof_dashboard():
    return render_template('skyroof.html',
                           obs_lat=OBS_LAT, obs_lon=OBS_LON, obs_name=OBS_NAME)


@app.route('/api/skyroof')
def skyroof_data():
    log_path    = request.args.get('log',     LOG_PATH)
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


@app.route('/api/config')
def config():
    return jsonify({
        'obs_lat':  OBS_LAT,
        'obs_lon':  OBS_LON,
        'obs_name': OBS_NAME,
    })


@app.route('/night')
@app.route('/night/<date_str>')
def night_view(date_str=None):
    return render_template('night.html', default_date=date_str or '')


@app.route('/api/night/<date_str>')
def night_data_route(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date, use YYYY-MM-DD'}), 400

    data    = get_night_data(date_str)
    actions = read_action_log(ACTION_LOG_PATH)

    start_s = datetime.fromtimestamp(data['window_start'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    end_s   = datetime.fromtimestamp(data['window_end']   / 1000).strftime('%Y-%m-%d %H:%M:%S')
    data['actions'] = [a for a in actions if start_s <= a['timestamp'] <= end_s]

    # Backfill sun times from OWM cache if darkness-flag detection didn't find them
    if not data['sunset_ts'] or not data['sunrise_ts']:
        forecast = get_forecast('virginia')   # uses cached data, no extra API call
        if forecast:
            for st in forecast.get('sun_times', []):
                if st['date'] == date_str:
                    data['sunset_ts']  = data['sunset_ts']  or st['sunset_ts']
                    data['sunrise_ts'] = data['sunrise_ts'] or st['sunrise_ts']
                    break

    # Compute and persist nightly stats for completed nights
    compute_and_store_nightly_stats(date_str)

    return jsonify(data)


@app.route('/calendar')
def calendar_view():
    return render_template('calendar.html', obs_name=OBS_NAME)


@app.route('/api/calendar')
def calendar_data():
    return jsonify(get_calendar_stats())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
