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
                alert        INTEGER,
                darkness     INTEGER
            )
        ''')
        # nightly_stats is never purged — grows permanently for the yearly calendar
        conn.execute('''
            CREATE TABLE IF NOT EXISTS nightly_stats (
                date        TEXT PRIMARY KEY,  -- YYYY-MM-DD (evening date)
                hours_open  REAL,
                hours_total REAL,
                cloud_avg   REAL,
                temp_avg    REAL,
                wind_avg    REAL,
                updated_at  TEXT,
                sunset_ts   INTEGER,
                sunrise_ts  INTEGER
            )
        ''')
    _migrate_db()


def _migrate_db():
    """Add columns introduced after initial schema deployment to existing DBs."""
    if not os.path.isfile(DB_PATH):
        return
    try:
        with _connect() as conn:
            wr_cols = {row[1] for row in conn.execute('PRAGMA table_info(weather_readings)').fetchall()}
            if 'darkness' not in wr_cols:
                conn.execute('ALTER TABLE weather_readings ADD COLUMN darkness INTEGER')

            ns_cols = {row[1] for row in conn.execute('PRAGMA table_info(nightly_stats)').fetchall()}
            if 'sunset_ts' not in ns_cols:
                conn.execute('ALTER TABLE nightly_stats ADD COLUMN sunset_ts INTEGER')
            if 'sunrise_ts' not in ns_cols:
                conn.execute('ALTER TABLE nightly_stats ADD COLUMN sunrise_ts INTEGER')
    except sqlite3.OperationalError:
        pass  # table doesn't exist yet; CREATE TABLE in init_db handles it


def save_reading(entry: dict):
    init_db()
    cutoff = (datetime.now() - timedelta(hours=RETENTION_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
    with _connect() as conn:
        conn.execute('''
            INSERT OR IGNORE INTO weather_readings
                (timestamp, ambient_temp, sky_temp, humidity, dew_point,
                 wind_speed, cloud_flag, rain_cond, roof, alert, darkness)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            entry.get('darkness'),
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

    # Fall back to previously persisted sun times (from an earlier compute or OWM backfill)
    if (not sunset_ts or not sunrise_ts) and os.path.isfile(DB_PATH):
        try:
            with _connect() as conn:
                stored = conn.execute(
                    'SELECT sunset_ts, sunrise_ts FROM nightly_stats WHERE date = ?',
                    (date_str,)
                ).fetchone()
            if stored:
                sunset_ts  = sunset_ts  or stored['sunset_ts']
                sunrise_ts = sunrise_ts or stored['sunrise_ts']
        except Exception:
            pass

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


# ── Nightly stats (permanent, never purged) ───────────────────────────────────

INTERVAL_S = 20  # SkyRoof writes every ~20 seconds


def _hours_open_from_actions(
    actions: list[dict],
    window_start_ms: int,
    window_end_ms: int,
) -> float | None:
    """Compute roof open time from action event timestamps.

    More accurate than row-counting because it handles data gaps (app restarts,
    log delays) that cause row-counting to under-report.
    Returns None if there are no usable action events inside the window.
    """
    if not actions:
        return None

    win_start   = datetime.fromtimestamp(window_start_ms / 1000)
    win_end     = datetime.fromtimestamp(window_end_ms   / 1000)
    win_start_s = win_start.strftime('%Y-%m-%d %H:%M:%S')
    win_end_s   = win_end.strftime('%Y-%m-%d %H:%M:%S')

    in_window = sorted(
        [a for a in actions if win_start_s <= a['timestamp'] <= win_end_s],
        key=lambda a: a['timestamp'],
    )
    if not in_window:
        return None

    CLOSE_TYPES  = {'close', 'cloud_close', 'rain_close', 'wind_close', 'scheduled_close'}
    open_start   = None
    open_seconds = 0.0

    for a in in_window:
        ts = datetime.strptime(a['timestamp'][:19], '%Y-%m-%d %H:%M:%S')
        if a['action_type'] == 'open':
            open_start = ts
        elif a['action_type'] in CLOSE_TYPES and open_start is not None:
            open_seconds += (ts - open_start).total_seconds()
            open_start = None

    # Roof still open at end of window
    if open_start is not None:
        open_seconds += (win_end - open_start).total_seconds()

    return max(0.0, open_seconds / 3600)


def compute_and_store_nightly_stats(
    date_str: str,
    actions: list[dict] | None = None,
    sunset_ts: int | None = None,
    sunrise_ts: int | None = None,
) -> dict | None:
    """Compute stats for a completed night and upsert into nightly_stats.
    Returns None if the night window hasn't ended yet or has no data.

    Parameters
    ----------
    date_str   : Evening date YYYY-MM-DD.
    actions    : Roof action events for this night — used for accurate hours_open.
                 Falls back to row-counting when None or empty.
    sunset_ts  : Known sunset epoch-ms (e.g. from OWM backfill).  Stored so
                 future loads of this night can show the sunset marker without
                 re-fetching OWM.
    sunrise_ts : Known sunrise epoch-ms.
    """
    data = get_night_data(date_str)
    rows = data['rows']
    if not rows:
        return None

    window_end_dt = datetime.fromtimestamp(data['window_end'] / 1000)
    if window_end_dt > datetime.now():
        return None  # night not yet complete

    hours_total = len(rows) * INTERVAL_S / 3600

    # Prefer action-based open hours (accurate); fall back to row-counting
    hours_open_evt = _hours_open_from_actions(
        actions or [], data['window_start'], data['window_end']
    )
    if hours_open_evt is not None:
        hours_open = hours_open_evt
    else:
        open_rows  = sum(1 for r in rows if (r.get('roof') or '').lower() == 'open')
        hours_open = open_rows * INTERVAL_S / 3600

    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    # Sun times: prefer explicitly passed values (OWM-enriched), then darkness-detected
    final_sunset_ts  = sunset_ts  or data.get('sunset_ts')
    final_sunrise_ts = sunrise_ts or data.get('sunrise_ts')

    stats = {
        'date':        date_str,
        'hours_open':  round(hours_open, 2),
        'hours_total': round(hours_total, 2),
        'cloud_avg':   avg('cloud_approx_pct'),
        'temp_avg':    avg('ambient_temp'),
        'wind_avg':    avg('wind_speed'),
        'updated_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sunset_ts':   final_sunset_ts,
        'sunrise_ts':  final_sunrise_ts,
    }

    init_db()
    with _connect() as conn:
        # Never overwrite previously stored good sun times with None
        existing = conn.execute(
            'SELECT sunset_ts, sunrise_ts FROM nightly_stats WHERE date = ?', (date_str,)
        ).fetchone()
        if existing:
            if stats['sunset_ts'] is None:
                stats['sunset_ts'] = existing['sunset_ts']
            if stats['sunrise_ts'] is None:
                stats['sunrise_ts'] = existing['sunrise_ts']

        conn.execute('''
            INSERT OR REPLACE INTO nightly_stats
                (date, hours_open, hours_total, cloud_avg, temp_avg, wind_avg,
                 updated_at, sunset_ts, sunrise_ts)
            VALUES (:date, :hours_open, :hours_total, :cloud_avg, :temp_avg, :wind_avg,
                    :updated_at, :sunset_ts, :sunrise_ts)
        ''', stats)

    return stats


def get_calendar_stats() -> list[dict]:
    """Return all nightly stats ordered by date for the calendar heatmap."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            'SELECT date, hours_open, hours_total, cloud_avg, temp_avg, wind_avg FROM nightly_stats ORDER BY date ASC'
        ).fetchall()
    return [dict(r) for r in rows]
