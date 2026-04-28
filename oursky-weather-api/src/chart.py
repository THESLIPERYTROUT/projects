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
        f'https://api.openweathermap.org/data/3.0/onecall'
        f'?lat={lat}&lon={lon}'
        f'&exclude=minutely,current,alerts'
        f'&units=imperial'   # °F / mph to match SkyRoof display
        f'&appid={OWM_KEY}'
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None

    data   = resp.json()
    hourly = data.get('hourly', [])
    daily  = data.get('daily',  [])

    forecast = [
        {
            'ts':         h['dt'] * 1000,
            'temp_f':     round(h['temp'], 1),
            'cloud_pct':  h['clouds'],
            'wind_mph':   round(h['wind_speed'], 1),
            'precip_pct': round(h.get('pop', 0) * 100),
        }
        for h in hourly
    ]

    # Night bands: sunset[N] → sunrise[N+1]
    night_bands = []
    for i, day in enumerate(daily):
        # Pre-sunrise darkness on the first day (stretches back into history)
        if i == 0:
            night_bands.append({
                'start': (day['dt'] - 43200) * 1000,   # ~midnight before
                'end':   day['sunrise'] * 1000,
            })
        if i + 1 < len(daily):
            night_bands.append({
                'start': day['sunset'] * 1000,
                'end':   daily[i + 1]['sunrise'] * 1000,
            })

    return {
        'forecast':    forecast,
        'night_bands': night_bands,
        'now_ts':      int(datetime.now(timezone.utc).timestamp() * 1000),
    }
