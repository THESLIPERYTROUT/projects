import sqlite3
import os
import time
from datetime import datetime, timedelta

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'weather_history.db'
)
RETENTION_HOURS = 168   # 7 days — matches the OWM 8-day forecast window
CLOUD_PCT = {1: 5, 2: 45, 3: 88}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS weather_readings (
                timestamp    TEXT PRIMARY KEY,
                ambient_temp REAL,
                sky_temp     REAL,
                humidity     REAL,
                dew_point    REAL,
                wind_speed   REAL,
                cloud_flag   INTEGER,
                rain_cond    INTEGER,
                roof         TEXT,
                alert        INTEGER
            )
        ''')


def save_reading(entry: dict):
    init_db()
    cutoff = (datetime.now() - timedelta(hours=RETENTION_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
    with _connect() as conn:
        conn.execute('''
            INSERT OR IGNORE INTO weather_readings
                (timestamp, ambient_temp, sky_temp, humidity, dew_point,
                 wind_speed, cloud_flag, rain_cond, roof, alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.get('timestamp'),
            entry.get('ambient_temp'),
            entry.get('sky_temp'),
            entry.get('humidity'),
            entry.get('dew_point'),
            entry.get('wind_speed'),
            entry.get('cloud_flag'),
            entry.get('rain_cond'),
            entry.get('roof'),
            int(bool(entry.get('alert'))),
        ))
        conn.execute('DELETE FROM weather_readings WHERE timestamp < ?', (cutoff,))


def _ts_to_ms(ts_str: str) -> int:
    """Local-time string 'YYYY-MM-DD HH:MM:SS[.xx]' → UTC milliseconds."""
    dt = datetime.strptime(ts_str[:19], '%Y-%m-%d %H:%M:%S')
    return int(time.mktime(dt.timetuple()) * 1000)


def get_night_data(date_str: str) -> dict:
    """Return all readings and detected sun times for a single night.

    'date_str' is the evening date (YYYY-MM-DD).
    Window: 4pm that day → 10am next day.
    """
    from datetime import timedelta
    base   = datetime.strptime(date_str, '%Y-%m-%d')
    start  = base.replace(hour=16, minute=0, second=0)
    end    = (base + timedelta(days=1)).replace(hour=10, minute=0, second=0)

    rows = []
    if os.path.isfile(DB_PATH):
        with _connect() as conn:
            raw = conn.execute(
                'SELECT * FROM weather_readings WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC',
                (start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'))
            ).fetchall()
        for r in raw:
            d = dict(r)
            d['ts'] = _ts_to_ms(d['timestamp'])
            d['cloud_approx_pct'] = CLOUD_PCT.get(d.get('cloud_flag'))
            rows.append(d)

    # Detect sunset/sunrise from darkness flag transitions (3=Daylight, 1/2=Dark/Dim)
    sunset_ts = sunrise_ts = None
    for i in range(1, len(rows)):
        prev_d = rows[i - 1].get('darkness')
        curr_d = rows[i].get('darkness')
        if prev_d == 3 and curr_d in (1, 2) and sunset_ts is None:
            sunset_ts = rows[i]['ts']
        if prev_d in (1, 2) and curr_d == 3 and sunrise_ts is None:
            sunrise_ts = rows[i]['ts']

    return {
        'date':         date_str,
        'window_start': int(start.timestamp() * 1000),
        'window_end':   int(end.timestamp() * 1000),
        'sunset_ts':    sunset_ts,
        'sunrise_ts':   sunrise_ts,
        'rows':         rows,
    }


def get_history_for_chart(hours: int = 48) -> list[dict]:
    if not os.path.isfile(DB_PATH):
        return []
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
    with _connect() as conn:
        rows = conn.execute(
            'SELECT * FROM weather_readings WHERE timestamp >= ? ORDER BY timestamp ASC',
            (cutoff,)
        ).fetchall()

    rows = [dict(r) for r in rows]

    # Downsample to ~1500 points max — enough for ~7 min resolution over 7 days
    step = max(1, len(rows) // 1500)
    rows = rows[::step]

    for r in rows:
        r['ts'] = _ts_to_ms(r['timestamp'])
        cf = r.get('cloud_flag')
        r['cloud_approx_pct'] = CLOUD_PCT.get(cf)

    return rows
