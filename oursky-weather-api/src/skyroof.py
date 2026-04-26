import re
import os
from datetime import datetime

LOG_PATH = os.environ.get(
    'SKYROOF_LOG_PATH',
    r'C:\Users\oursky\Documents\Interactiveastronomy\SkyRoof\weatherfilelog.txt'
)
ACTION_LOG_PATH = os.environ.get(
    'SKYROOF_ACTION_LOG_PATH',
    r'C:\Users\oursky\Documents\Interactiveastronomy\SkyRoof\skyroof_log.txt'
)
MAX_HISTORY = 20
MAX_ACTIONS = 30

# Matches both the one-line data file and the multi-entry log (which prefixes with "N)  " and
# appends "Status:[...] Scope=... Roof=...")
LINE_RE = re.compile(
    r'(?:^\s*\d+\)\s+)?'                  # optional index (log file only)
    r'(\d{4}-\d{2}-\d{2})\s+'             # date
    r'(\d{2}:\d{2}:\d{2}\.\d+)\s+'        # time
    r'([CF])\s+'                           # temp_scale  (C/F)
    r'([MK])\s+'                           # wind_scale  (M=Mph, K=Knots)
    r'([-\d.]+)\s+'                        # sky_temp
    r'([-\d.]+)\s+'                        # ambient_temp
    r'([-\d.]+)\s+'                        # sensor_temp
    r'([\d.]+)\s+'                         # wind_speed
    r'([\d.]+)\s+'                         # humidity
    r'([-\d.]+)\s+'                        # dew_point
    r'(\d+)\s+'                            # dew_heater_pct
    r'(\d+)\s+'                            # rain_flag   (raw)
    r'(\d+)\s+'                            # wet_flag    (raw)
    r'(\d+)\s+'                            # elapsed_sec
    r'([\d.]+)\s+'                         # elapsed_days (Julian)
    r'([123])\s+'                          # cloud_flag  (1=Clear,2=Light Clouds,3=Very Cloudy)
    r'([123])\s+'                          # wind_flag   (1=Calm,2=Windy,3=Very Windy)
    r'([123])\s+'                          # rain_cond   (1=Dry,2=Damp,3=Rain)
    r'([123])\s+'                          # darkness    (1=Dark,2=Dim,3=Daylight)
    r'(\d+)\s+'                            # roof_close_flag
    r'(\d+)'                               # alert_flag  (0=OK,1=Alert)
    r'(?:\s+Status:\[([^\]]+)\]'           # status      (log only)
    r'\s+Scope=(\S+)'                      # scope       (log only)
    r'\s+Roof=(\S+))?',                    # roof        (log only)
    re.IGNORECASE | re.MULTILINE
)

CLOUD_LABELS  = {'1': 'Clear', '2': 'Light Clouds', '3': 'Very Cloudy'}
WIND_LABELS   = {'1': 'Calm',  '2': 'Windy',        '3': 'Very Windy'}
RAIN_LABELS   = {'1': 'Dry',   '2': 'Damp',         '3': 'Rain'}
DARK_LABELS   = {'1': 'Dark',  '2': 'Dim',           '3': 'Daylight'}


def _to_c(f_val: float) -> float:
    return round((f_val - 32) * 5 / 9, 1)


def parse_line(line: str) -> dict | None:
    m = LINE_RE.search(line)
    if not m:
        return None

    (date, time_, temp_scale, wind_scale,
     sky_temp, ambient_temp, sensor_temp,
     wind_speed, humidity, dew_point,
     dew_heater, rain_flag_raw, wet_flag_raw,
     elapsed_sec, elapsed_days,
     cloud_flag, wind_flag, rain_cond, darkness,
     roof_close_flag, alert_flag,
     status, scope, roof) = m.groups()

    fahrenheit = temp_scale.upper() == 'F'
    mph = wind_scale.upper() == 'M'

    def maybe_c(v):
        return _to_c(float(v)) if fahrenheit else float(v)

    ambient = float(ambient_temp)
    sky     = float(sky_temp)
    dew     = float(dew_point)

    return {
        'timestamp':        f'{date} {time_}',
        'date':             date,
        'time':             time_[:8],
        'temp_scale':       temp_scale.upper(),
        'wind_scale':       'Mph' if mph else 'Knots',

        'sky_temp':         sky,
        'sky_temp_c':       maybe_c(sky_temp) if fahrenheit else sky,
        'ambient_temp':     ambient,
        'ambient_temp_c':   maybe_c(ambient_temp) if fahrenheit else ambient,
        'sensor_temp':      float(sensor_temp),

        'wind_speed':       float(wind_speed),
        'humidity':         float(humidity),
        'dew_point':        dew,
        'dew_point_c':      maybe_c(dew_point) if fahrenheit else dew,
        'dew_heater_pct':   int(dew_heater),

        'rain_flag_raw':    int(rain_flag_raw),
        'wet_flag_raw':     int(wet_flag_raw),
        'elapsed_sec':      int(elapsed_sec),

        'cloud_flag':       int(cloud_flag),
        'cloud_label':      CLOUD_LABELS.get(cloud_flag, cloud_flag),
        'wind_flag':        int(wind_flag),
        'wind_label':       WIND_LABELS.get(wind_flag, wind_flag),
        'rain_cond':        int(rain_cond),
        'rain_label':       RAIN_LABELS.get(rain_cond, rain_cond),
        'darkness':         int(darkness),
        'darkness_label':   DARK_LABELS.get(darkness, darkness),

        'roof_close_flag':  int(roof_close_flag),
        'alert':            int(alert_flag) == 1,
        'alert_flag':       int(alert_flag),

        # log-only fields (None when reading the one-line data file)
        'status':           status,
        'scope':            scope,
        'roof':             roof,
    }


def read_log(path: str = LOG_PATH, max_lines: int = MAX_HISTORY) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [l for l in f.readlines() if l.strip()]
    parsed = []
    for line in reversed(lines):
        entry = parse_line(line)
        if entry:
            parsed.append(entry)
            if len(parsed) >= max_lines:
                break
    return parsed


def latest(path: str = LOG_PATH) -> dict | None:
    entries = read_log(path, max_lines=1)
    return entries[0] if entries else None


# ── Roof action log (skyroof_log.txt) ────────────────────────────────────────

ACTION_RE = re.compile(
    r'^\s*(\d{2}-\d{2}-\d{4})\s+'   # date MM-DD-YYYY
    r'(\d{1,2}:\d{2}:\d{2}\s+[AP]M)'  # time H:MM:SS AM/PM
    r':\s+(.+)$',                    # message
    re.IGNORECASE
)

# Keywords → (action_type, display label)
_ACTION_PATTERNS = [
    (r'did not open.*unsafe',           'blocked',    'Did not open — unsafe conditions'),
    (r'did not open',                   'blocked',    'Did not open'),
    (r'dusk.?dawn.*closed|closed.*dusk.?dawn', 'scheduled_close', 'Closed — scheduled dusk/dawn'),
    (r'closed.*cloudiness|cloudiness',  'cloud_close','Closed — excessive cloudiness'),
    (r'closed.*rain|rain.*closed',      'rain_close', 'Closed — rain detected'),
    (r'closed.*wind|wind.*closed',      'wind_close', 'Closed — high wind'),
    (r'opened|re-opened|re.opened',     'open',       'Opened'),
    (r'closed',                         'close',      'Closed'),
]

def _classify_action(message: str) -> tuple[str, str]:
    msg_lower = message.lower()
    for pattern, atype, label in _ACTION_PATTERNS:
        if re.search(pattern, msg_lower):
            return atype, label
    return 'unknown', message.strip()


def parse_action_line(line: str) -> dict | None:
    m = ACTION_RE.match(line.strip())
    if not m:
        return None
    date_str, time_str, message = m.groups()
    action_type, action_label = _classify_action(message)
    try:
        dt = datetime.strptime(f'{date_str} {time_str.strip()}', '%m-%d-%Y %I:%M:%S %p')
        timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        date_fmt  = dt.strftime('%b %d, %Y')
        time_fmt  = dt.strftime('%I:%M %p').lstrip('0')
    except ValueError:
        timestamp = f'{date_str} {time_str.strip()}'
        date_fmt  = date_str
        time_fmt  = time_str.strip()
    return {
        'timestamp':    timestamp,
        'date':         date_fmt,
        'time':         time_fmt,
        'message':      message.strip(),
        'action_type':  action_type,
        'action_label': action_label,
    }


def read_action_log(path: str = ACTION_LOG_PATH, max_lines: int = MAX_ACTIONS) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [l for l in f.readlines() if l.strip()]
    parsed = []
    for line in reversed(lines):
        entry = parse_action_line(line)
        if entry:
            parsed.append(entry)
            if len(parsed) >= max_lines:
                break
    return parsed
