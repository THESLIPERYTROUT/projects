import requests
from datetime import datetime, timezone
import time

OWM_KEY = 'db98887c21d08c99b463ba97957717a4'

CACHE_TTL = 1800  # 30 minutes → 48 calls/day per site, well under the 1,000 limit
_cache: dict[str, tuple[float, dict]] = {}  # site → (fetched_at, data)

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
    key = site.lower()
    coords = SITES.get(key)
    if not coords:
        return None

    # Return cached result if still fresh
    if key in _cache:
        fetched_at, cached_data = _cache[key]
        if time.monotonic() - fetched_at < CACHE_TTL:
            # Update now_ts so the "Now" line stays accurate without a new OWM call
            cached_data['now_ts'] = int(datetime.now(timezone.utc).timestamp() * 1000)
            return cached_data

    lat, lon = coords
    url = (
        f'https://api.openweathermap.org/data/3.0/onecall'
        f'?lat={lat}&lon={lon}'
        f'&exclude=minutely,current,alerts'
        f'&units=imperial'
        f'&appid={OWM_KEY}'
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException:
        return _cache.get(key, (None, None))[1]  # serve stale on network error
    if resp.status_code != 200:
        print(f'OWM error {resp.status_code}: {resp.text[:200]}')
        return _cache.get(key, (None, None))[1]

    data   = resp.json()
    hourly = data.get('hourly', [])
    daily  = data.get('daily',  [])

    # Hourly forecast — first 48 h, full resolution
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

    # Daily forecast — days 3-8 (days 1-2 already covered by hourly above)
    daily_forecast = [
        {
            'ts':         d['dt'] * 1000,
            'temp_f':     round(d['temp']['day'], 1),
            'cloud_pct':  d.get('clouds', 0),
            'wind_mph':   round(d['wind_speed'], 1),
            'precip_pct': round(d.get('pop', 0) * 100),
        }
        for d in daily[2:]   # skip days 0-1 which overlap with hourly
    ]

    # Sun times per day — used by the night detail view as fallback
    sun_times = [
        {'date': datetime.fromtimestamp(d['dt'], tz=timezone.utc).strftime('%Y-%m-%d'),
         'sunrise_ts': d['sunrise'] * 1000,
         'sunset_ts':  d['sunset']  * 1000}
        for d in daily
    ]

    # Night bands: sunset[N] → sunrise[N+1]
    night_bands = []
    for i, day in enumerate(daily):
        if i == 0:
            night_bands.append({
                'start': (day['dt'] - 43200) * 1000,
                'end':   day['sunrise'] * 1000,
            })
        if i + 1 < len(daily):
            night_bands.append({
                'start': day['sunset'] * 1000,
                'end':   daily[i + 1]['sunrise'] * 1000,
            })

    result = {
        'forecast':       forecast,
        'daily_forecast': daily_forecast,
        'sun_times':      sun_times,
        'night_bands':    night_bands,
        'now_ts':         int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    _cache[key] = (time.monotonic(), result)
    return result
