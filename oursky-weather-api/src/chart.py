import requests
from datetime import datetime, timezone

OWM_KEY = 'db98887c21d08c99b463ba97957717a4'

SITES = {
    'virginia':  (39.095472,  -77.919111),
    'namibia':   (-23.236390,  16.361670),
    'chile':     (-30.470556, -70.765000),
    'sro':       (37.070367,  -119.413093),
    'awoa':      (-31.820398,  117.281526),
    'wolongbar': (-28.820056,  153.420472),
    'brazil':    (-21.736300,  -41.026200),
}


def get_forecast(site: str) -> dict | None:
    coords = SITES.get(site.lower())
    if not coords:
        return None
    lat, lon = coords

    url = (
        f'https://api.openweathermap.org/data/2.5/forecast'
        f'?lat={lat}&lon={lon}'
        f'&units=imperial'
        f'&appid={OWM_KEY}'
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        print(f'OWM error {resp.status_code}: {resp.text[:200]}')
        return None

    data = resp.json()
    entries = data.get('list', [])
    city    = data.get('city', {})

    forecast = [
        {
            'ts':         e['dt'] * 1000,
            'temp_f':     round(e['main']['temp'], 1),
            'cloud_pct':  e['clouds']['all'],
            'wind_mph':   round(e['wind']['speed'], 1),
            'precip_pct': round(e.get('pop', 0) * 100),
        }
        for e in entries
    ]

    # Build ~5 days of night bands from the city's today sunrise/sunset,
    # stepping forward one day at a time.
    SIDEREAL = 86400  # seconds per day (close enough)
    base_sunrise = city.get('sunrise', 0)
    base_sunset  = city.get('sunset',  0)
    night_bands  = []
    for day_offset in range(-1, 6):
        sunrise = (base_sunrise + day_offset * SIDEREAL) * 1000
        sunset  = (base_sunset  + day_offset * SIDEREAL) * 1000
        next_sunrise = sunrise + SIDEREAL * 1000
        night_bands.append({'start': sunset, 'end': next_sunrise})

    return {
        'forecast':    forecast,
        'night_bands': night_bands,
        'now_ts':      int(datetime.now(timezone.utc).timestamp() * 1000),
    }
