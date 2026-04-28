import sqlite3
import os
import time
from datetime import datetime, timedelta

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'weather_history.db'
)
RETENTION_HOURS = 48
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

    # Downsample to ~500 points max so Chart.js stays responsive
    step = max(1, len(rows) // 500)
    rows = rows[::step]

    for r in rows:
        r['ts'] = _ts_to_ms(r['timestamp'])
        cf = r.get('cloud_flag')
        r['cloud_approx_pct'] = CLOUD_PCT.get(cf)

    return rows
